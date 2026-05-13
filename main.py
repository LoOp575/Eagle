"""
EAGLE - Crypto Futures Intelligence System.

Main async orchestrator that coordinates all system components:
market scanning, feature engineering, scoring, risk management,
trade execution, and alert dispatch.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from typing import List, Optional

from dotenv import load_dotenv

from alerts.alert_manager import AlertManager
from alerts.console_dashboard import ConsoleDashboard
from config import SystemConfig, load_config
from execution.trade_executor import TradeExecutor
from features.feature_engine import FeatureEngine
from features.feature_vector import FeatureVector
from risk_engine.position_manager import PositionManager
from risk_engine.risk_controls import RiskController
from scanner.data_models import MarketData
from scanner.market_scanner import MarketScanner
from scoring.decision_engine import DecisionEngine
from scoring.dump_score import DumpScorer
from scoring.pump_score import PumpScorer
from scoring.signal_models import SignalType, TradeSignal

logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
_shutdown_requested: bool = False


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed argument namespace with mode, exchange, interval, and config options.
    """
    parser = argparse.ArgumentParser(
        prog="eagle",
        description="EAGLE - Crypto Futures Intelligence System",
    )
    parser.add_argument(
        "--mode",
        choices=["scan-only", "full-auto", "monitor"],
        default="scan-only",
        help="Operating mode: scan-only (default), full-auto (with execution), monitor (dashboard only)",
    )
    parser.add_argument(
        "--exchange",
        choices=["binance", "bybit"],
        default="binance",
        help="Exchange to connect to (default: binance)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Scan interval in seconds (default: 60)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration override file",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def setup_signal_handlers() -> None:
    """Register SIGINT and SIGTERM handlers for graceful shutdown."""
    global _shutdown_requested

    def _handle_shutdown(signum: int, frame: object) -> None:
        global _shutdown_requested
        logger.info(
            "Shutdown signal received (signal %d). Shutting down gracefully...",
            signum,
        )
        _shutdown_requested = True

    signal.signal(signal.SIGINT, _handle_shutdown)
    signal.signal(signal.SIGTERM, _handle_shutdown)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging for the application.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
    """
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


async def main() -> None:
    """Main async orchestrator for the EAGLE system.

    Initializes all components and runs the main scanning loop
    with proper error handling and graceful shutdown support.
    """
    global _shutdown_requested

    load_dotenv()

    args = parse_arguments()
    setup_logging(args.log_level)
    setup_signal_handlers()

    # Load configuration
    config = load_config()

    # Override exchange from CLI
    config.exchange.exchange = args.exchange

    # Display startup banner
    dashboard = ConsoleDashboard(config.alerts)
    dashboard.display_startup_banner()

    logger.info("Starting EAGLE in '%s' mode on %s", args.mode, args.exchange)
    logger.info("Scan interval: %d seconds", args.interval)

    # Initialize components
    scanner = MarketScanner(config)
    feature_engine = FeatureEngine(config.indicators)
    pump_scorer = PumpScorer(config.scoring)
    dump_scorer = DumpScorer(config.scoring)
    decision_engine = DecisionEngine(config.scoring, config.trading)
    position_manager = PositionManager(config.risk)
    risk_controller = RiskController(config.risk)
    alert_manager = AlertManager(config.alerts)

    # Initialize trade executor only in full-auto mode
    trade_executor: Optional[TradeExecutor] = None
    if args.mode == "full-auto" and config.trading.trading_enabled:
        trade_executor = TradeExecutor(
            exchange_config=config.exchange,
            trading_config=config.trading,
            position_manager=position_manager,
            risk_controller=risk_controller,
        )

    try:
        # Initialize exchange connection
        await scanner.initialize()
        logger.info("Scanner initialized successfully")

        if trade_executor is not None:
            await trade_executor.initialize()
            logger.info("Trade executor initialized")

        # Main scanning loop
        while not _shutdown_requested:
            cycle_start = time.time()
            logger.info("--- Scan cycle starting ---")

            try:
                signals = await _run_scan_cycle(
                    scanner=scanner,
                    feature_engine=feature_engine,
                    pump_scorer=pump_scorer,
                    dump_scorer=dump_scorer,
                    decision_engine=decision_engine,
                )

                # Sort signals by pump_score descending
                signals.sort(key=lambda s: s.pump_score, reverse=True)

                # Filter actionable signals (score > 50)
                actionable = [
                    s for s in signals
                    if s.pump_score > 50.0 or s.dump_score > 50.0
                ]

                # Display market overview
                signals_by_type: dict = {}
                for s in signals:
                    stype = s.signal_type.value
                    signals_by_type[stype] = signals_by_type.get(stype, 0) + 1
                dashboard.display_market_overview(len(signals), signals_by_type)

                # Dispatch alerts for actionable signals
                await alert_manager.dispatch_batch(actionable)

                # Execute trades in full-auto mode
                if (
                    args.mode == "full-auto"
                    and trade_executor is not None
                    and config.trading.trading_enabled
                ):
                    await _execute_top_signals(
                        signals=actionable,
                        trade_executor=trade_executor,
                        risk_controller=risk_controller,
                        position_manager=position_manager,
                    )

                # Log cycle summary
                cycle_duration = time.time() - cycle_start
                logger.info(
                    "Scan cycle complete: %d symbols, %d actionable signals, "
                    "%.1fs duration",
                    len(signals),
                    len(actionable),
                    cycle_duration,
                )

            except Exception as e:
                logger.error("Error in scan cycle: %s", str(e), exc_info=True)

            # Sleep until next cycle (check shutdown frequently)
            sleep_end = time.time() + args.interval
            while time.time() < sleep_end and not _shutdown_requested:
                await asyncio.sleep(1.0)

    except Exception as e:
        logger.critical("Fatal error in main loop: %s", str(e), exc_info=True)

    finally:
        # Graceful shutdown
        logger.info("Initiating graceful shutdown...")
        try:
            await scanner.close()
            logger.info("Scanner closed")
        except Exception as e:
            logger.error("Error closing scanner: %s", str(e))

        if trade_executor is not None:
            try:
                await trade_executor.close()
                logger.info("Trade executor closed")
            except Exception as e:
                logger.error("Error closing trade executor: %s", str(e))

        logger.info("EAGLE system shut down complete.")


async def _run_scan_cycle(
    scanner: MarketScanner,
    feature_engine: FeatureEngine,
    pump_scorer: PumpScorer,
    dump_scorer: DumpScorer,
    decision_engine: DecisionEngine,
) -> List[TradeSignal]:
    """Run a single scan cycle across all symbols.

    Fetches market data, computes features, scores, and generates signals.

    Args:
        scanner: Initialized MarketScanner.
        feature_engine: FeatureEngine for indicator computation.
        pump_scorer: PumpScorer for bullish probability.
        dump_scorer: DumpScorer for bearish probability.
        decision_engine: DecisionEngine for signal classification.

    Returns:
        List of TradeSignal for all scanned symbols.
    """
    signals: List[TradeSignal] = []

    # Fetch all futures symbols
    symbols = await scanner.fetch_all_futures_symbols()
    logger.info("Scanning %d futures symbols", len(symbols))

    # Scan all symbols in batches
    scan_results = await scanner.scan_all()

    # Collect all market data for attention ranking
    all_market_data: List[MarketData] = [
        result.market_data for result in scan_results if result.success
    ]

    # Process each scan result
    for result in scan_results:
        if not result.success:
            continue

        try:
            # Compute features
            feature_vector = feature_engine.compute_features(
                result.market_data, all_market_data=all_market_data
            )

            # Compute scores
            pump_score, pump_reasons = pump_scorer.compute_pump_score(feature_vector)
            dump_score, dump_reasons = dump_scorer.compute_dump_score(feature_vector)

            # Generate trade signal
            trade_signal = decision_engine.generate_signal(
                pump_score=pump_score,
                dump_score=dump_score,
                pump_reasons=pump_reasons,
                dump_reasons=dump_reasons,
                feature_vector=feature_vector,
            )

            signals.append(trade_signal)

        except Exception as e:
            logger.warning(
                "Failed to process %s: %s", result.symbol, str(e)
            )

    return signals


async def _execute_top_signals(
    signals: List[TradeSignal],
    trade_executor: TradeExecutor,
    risk_controller: RiskController,
    position_manager: PositionManager,
) -> None:
    """Execute top-ranked signals that pass risk gates.

    Args:
        signals: Actionable signals sorted by pump_score descending.
        trade_executor: TradeExecutor instance.
        risk_controller: RiskController for risk gate checks.
        position_manager: PositionManager for position tracking.
    """
    if not risk_controller.should_trade():
        logger.info("Risk controller blocking all trades")
        return

    # Get account balance for position sizing
    account_balance = await trade_executor.get_account_balance()
    if account_balance <= 0.0:
        logger.warning("Could not fetch account balance, skipping execution")
        return

    for trade_signal in signals:
        # Only execute HIGH_PROBABILITY_LONG or HIGH_RISK_DUMP_SHORT
        if trade_signal.signal_type not in (
            SignalType.HIGH_PROBABILITY_LONG,
            SignalType.HIGH_RISK_DUMP_SHORT,
        ):
            continue

        # Check if we can still open positions
        if not position_manager.can_open_position():
            logger.info("Max positions reached, stopping execution")
            break

        # Check risk gate before each trade
        if not risk_controller.should_trade():
            logger.info("Risk gate closed, stopping execution")
            break

        try:
            result = await trade_executor.execute_signal(
                trade_signal, account_balance
            )
            if result is not None:
                logger.info(
                    "Trade executed: %s %s @ %s",
                    result["side"],
                    result["symbol"],
                    result["entry_price"],
                )
        except Exception as e:
            logger.error(
                "Trade execution failed for %s: %s",
                trade_signal.symbol,
                str(e),
            )


if __name__ == "__main__":
    asyncio.run(main())

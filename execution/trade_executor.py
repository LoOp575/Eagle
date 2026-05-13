"""
Trade execution module.

Handles async order placement, stop loss/take profit management,
position closing, and exchange interaction via CCXT.
Supports both Binance Futures and Bybit exchanges.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Dict, List, Optional

from config import ExchangeConfig, TradingConfig
from risk_engine.position_manager import PositionManager
from risk_engine.risk_controls import RiskController
from scoring.signal_models import SignalType, TradeSignal

logger = logging.getLogger(__name__)


class TradeExecutor:
    """Executes trading signals on the exchange.

    Handles the full lifecycle of trade execution including:
    - Risk checks before execution
    - Dynamic leverage calculation
    - Position sizing
    - Market order placement
    - Stop loss and take profit order placement
    - Position tracking and closing
    """

    def __init__(
        self,
        exchange_config: ExchangeConfig,
        trading_config: TradingConfig,
        position_manager: PositionManager,
        risk_controller: RiskController,
    ) -> None:
        """Initialize trade executor with dependencies.

        Args:
            exchange_config: Exchange connection configuration.
            trading_config: Trading parameters (SL/TP multipliers, etc).
            position_manager: Position tracking and sizing manager.
            risk_controller: Risk gate and leverage controller.
        """
        self._exchange_config = exchange_config
        self._trading_config = trading_config
        self._position_manager = position_manager
        self._risk_controller = risk_controller
        self._exchange = None
        self._max_retries = 3
        self._retry_delay = 1.0

    async def initialize(self) -> None:
        """Initialize exchange connection.

        Creates the CCXT exchange instance based on configuration.
        Should be called before any trading operations.
        """
        try:
            # Dynamic import of ccxt to allow py_compile without the package
            import ccxt.async_support as ccxt_async  # type: ignore

            exchange_id = self._exchange_config.exchange.lower()

            if exchange_id == "binance":
                self._exchange = ccxt_async.binance({
                    "apiKey": self._exchange_config.binance_api_key,
                    "secret": self._exchange_config.binance_api_secret,
                    "sandbox": self._exchange_config.sandbox_mode,
                    "options": {
                        "defaultType": "future",
                        "adjustForTimeDifference": True,
                    },
                    "enableRateLimit": True,
                    "rateLimit": self._exchange_config.rate_limit_ms,
                })
            elif exchange_id == "bybit":
                self._exchange = ccxt_async.bybit({
                    "apiKey": self._exchange_config.bybit_api_key,
                    "secret": self._exchange_config.bybit_api_secret,
                    "sandbox": self._exchange_config.sandbox_mode,
                    "options": {
                        "defaultType": "linear",
                    },
                    "enableRateLimit": True,
                    "rateLimit": self._exchange_config.rate_limit_ms,
                })
            else:
                raise ValueError(f"Unsupported exchange: {exchange_id}")

            logger.info("Exchange %s initialized successfully", exchange_id)

        except ImportError:
            logger.error("ccxt package not available - trading disabled")
            self._exchange = None
        except Exception as e:
            logger.error("Failed to initialize exchange: %s", str(e))
            self._exchange = None

    async def close(self) -> None:
        """Close exchange connection and clean up resources."""
        if self._exchange is not None:
            try:
                await self._exchange.close()
            except Exception as e:
                logger.warning("Error closing exchange: %s", str(e))
            finally:
                self._exchange = None

    async def execute_signal(
        self, signal: TradeSignal, account_balance: float
    ) -> Optional[dict]:
        """Execute a trade signal with full risk checks.

        Performs pre-trade validation, computes position size and leverage,
        places orders, and records the position.

        Args:
            signal: Trade signal to execute.
            account_balance: Current account balance in USD.

        Returns:
            Order details dict if executed, None if rejected or failed.
        """
        # Pre-trade risk checks
        if not self._risk_controller.should_trade():
            logger.warning(
                "Trade rejected by risk controller for %s", signal.symbol
            )
            return None

        if not self._position_manager.can_open_position():
            logger.warning(
                "Trade rejected: max positions reached for %s", signal.symbol
            )
            return None

        if self._position_manager.has_position(signal.symbol):
            logger.warning(
                "Trade rejected: position already open for %s", signal.symbol
            )
            return None

        if signal.signal_type == SignalType.NO_TRADE:
            return None

        if signal.signal_type == SignalType.EARLY_ACCUMULATION_WATCH:
            logger.info("Signal is watch-only for %s, not executing", signal.symbol)
            return None

        try:
            # Get current price for execution
            current_price = await self._get_current_price(signal.symbol)
            if current_price is None or current_price <= 0.0:
                logger.error("Could not get price for %s", signal.symbol)
                return None

            # Determine trade direction
            if signal.signal_type == SignalType.HIGH_PROBABILITY_LONG:
                side = "buy"
            elif signal.signal_type == SignalType.HIGH_RISK_DUMP_SHORT:
                side = "sell"
            else:
                return None

            # Compute dynamic leverage
            atr_ratio = 0.02  # Default if not available from signal
            if signal.suggested_stop_loss is not None:
                atr_ratio = abs(signal.suggested_stop_loss)
            leverage = self._risk_controller.compute_dynamic_leverage(atr_ratio)

            # Compute stop loss and take profit prices
            sl_distance = (
                atr_ratio * self._trading_config.stop_loss_atr_multiplier
            )
            tp_distance = (
                atr_ratio * self._trading_config.take_profit_atr_multiplier
            )

            if side == "buy":
                stop_loss_price = current_price * (1.0 - sl_distance)
                take_profit_price = current_price * (1.0 + tp_distance)
            else:
                stop_loss_price = current_price * (1.0 + sl_distance)
                take_profit_price = current_price * (1.0 - tp_distance)

            # Calculate position size
            position_size = self._position_manager.calculate_position_size(
                account_balance=account_balance,
                entry_price=current_price,
                stop_loss_price=stop_loss_price,
                leverage=leverage,
            )

            if position_size <= 0.0:
                logger.warning("Calculated position size is zero for %s", signal.symbol)
                return None

            # Set leverage on exchange
            await self._set_leverage(signal.symbol, leverage)

            # Place market order
            order = await self._place_market_order(
                signal.symbol, side, position_size, leverage
            )

            if order is None:
                return None

            # Place stop loss
            sl_side = "sell" if side == "buy" else "buy"
            sl_order = await self._place_stop_loss(
                signal.symbol, sl_side, position_size, stop_loss_price
            )

            # Place take profit
            tp_order = await self._place_take_profit(
                signal.symbol, sl_side, position_size, take_profit_price
            )

            # Record position
            self._position_manager.open_position(
                symbol=signal.symbol,
                entry_price=current_price,
                size=position_size,
                leverage=leverage,
                stop_loss=stop_loss_price,
                take_profit=take_profit_price,
            )

            result = {
                "symbol": signal.symbol,
                "side": side,
                "size": position_size,
                "leverage": leverage,
                "entry_price": current_price,
                "stop_loss": stop_loss_price,
                "take_profit": take_profit_price,
                "order_id": order.get("id"),
                "sl_order_id": sl_order.get("id") if sl_order else None,
                "tp_order_id": tp_order.get("id") if tp_order else None,
                "timestamp": time.time(),
                "signal_type": signal.signal_type.value,
                "confidence": signal.confidence.value,
            }

            logger.info(
                "Trade executed: %s %s %s @ %s (leverage: %sx)",
                side.upper(),
                position_size,
                signal.symbol,
                current_price,
                leverage,
            )

            return result

        except Exception as e:
            logger.error(
                "Trade execution failed for %s: %s", signal.symbol, str(e)
            )
            return None

    async def _place_market_order(
        self, symbol: str, side: str, amount: float, leverage: int
    ) -> Optional[dict]:
        """Place a market order with retry logic.

        Args:
            symbol: Trading pair symbol.
            side: Order side ('buy' or 'sell').
            amount: Order amount in base asset.
            leverage: Leverage for the position.

        Returns:
            Order response dict, or None on failure.
        """
        for attempt in range(self._max_retries):
            try:
                if self._exchange is None:
                    logger.error("Exchange not initialized")
                    return None

                order = await self._exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=side,
                    amount=amount,
                )
                logger.info(
                    "Market order placed: %s %s %s (attempt %d)",
                    side, amount, symbol, attempt + 1,
                )
                return order

            except Exception as e:
                logger.warning(
                    "Market order attempt %d failed for %s: %s",
                    attempt + 1, symbol, str(e),
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))

        logger.error("All market order attempts failed for %s", symbol)
        return None

    async def _place_stop_loss(
        self, symbol: str, side: str, amount: float, stop_price: float
    ) -> Optional[dict]:
        """Place a stop loss order.

        Args:
            symbol: Trading pair symbol.
            side: Close side ('sell' for long SL, 'buy' for short SL).
            amount: Position size to close.
            stop_price: Trigger price for stop loss.

        Returns:
            Order response dict, or None on failure.
        """
        for attempt in range(self._max_retries):
            try:
                if self._exchange is None:
                    return None

                order = await self._exchange.create_order(
                    symbol=symbol,
                    type="stop_market",
                    side=side,
                    amount=amount,
                    price=None,
                    params={"stopPrice": stop_price, "reduceOnly": True},
                )
                logger.info("Stop loss placed for %s at %s", symbol, stop_price)
                return order

            except Exception as e:
                logger.warning(
                    "Stop loss attempt %d failed for %s: %s",
                    attempt + 1, symbol, str(e),
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))

        logger.error("All stop loss attempts failed for %s", symbol)
        return None

    async def _place_take_profit(
        self, symbol: str, side: str, amount: float, tp_price: float
    ) -> Optional[dict]:
        """Place a take profit order.

        Args:
            symbol: Trading pair symbol.
            side: Close side ('sell' for long TP, 'buy' for short TP).
            amount: Position size to close.
            tp_price: Trigger price for take profit.

        Returns:
            Order response dict, or None on failure.
        """
        for attempt in range(self._max_retries):
            try:
                if self._exchange is None:
                    return None

                order = await self._exchange.create_order(
                    symbol=symbol,
                    type="take_profit_market",
                    side=side,
                    amount=amount,
                    price=None,
                    params={"stopPrice": tp_price, "reduceOnly": True},
                )
                logger.info("Take profit placed for %s at %s", symbol, tp_price)
                return order

            except Exception as e:
                logger.warning(
                    "Take profit attempt %d failed for %s: %s",
                    attempt + 1, symbol, str(e),
                )
                if attempt < self._max_retries - 1:
                    await asyncio.sleep(self._retry_delay * (attempt + 1))

        logger.error("All take profit attempts failed for %s", symbol)
        return None

    async def close_position(self, symbol: str) -> Optional[float]:
        """Close an open position by placing a market close order.

        Args:
            symbol: Trading pair symbol to close.

        Returns:
            Realized PnL, or None if no position or close failed.
        """
        position = self._position_manager.get_position(symbol)
        if position is None:
            logger.warning("No position found to close for %s", symbol)
            return None

        try:
            # Determine close side (opposite of entry)
            # If entry was buy, close with sell
            close_side = "sell"  # Default assumption for long
            # We can infer from stop loss vs entry
            if position.stop_loss > position.entry_price:
                # Stop loss above entry means this is a short position
                close_side = "buy"

            if self._exchange is not None:
                await self._exchange.create_order(
                    symbol=symbol,
                    type="market",
                    side=close_side,
                    amount=position.size,
                    params={"reduceOnly": True},
                )

            pnl = self._position_manager.close_position(symbol)
            self._risk_controller.record_trade_result(pnl if pnl is not None else 0.0)

            logger.info("Position closed for %s, PnL: %s", symbol, pnl)
            return pnl

        except Exception as e:
            logger.error("Failed to close position for %s: %s", symbol, str(e))
            return None

    async def close_all_positions(self) -> List[dict]:
        """Emergency close all open positions.

        Returns:
            List of close results with symbol and PnL.
        """
        results: List[dict] = []
        positions = self._position_manager.get_open_positions()

        for position in positions:
            try:
                pnl = await self.close_position(position.symbol)
                results.append({
                    "symbol": position.symbol,
                    "pnl": pnl,
                    "status": "closed" if pnl is not None else "failed",
                })
            except Exception as e:
                logger.error(
                    "Emergency close failed for %s: %s", position.symbol, str(e)
                )
                results.append({
                    "symbol": position.symbol,
                    "pnl": None,
                    "status": "failed",
                    "error": str(e),
                })

        return results

    async def get_account_balance(self) -> float:
        """Fetch current account balance from exchange.

        Returns:
            Account balance in USDT, or 0.0 on failure.
        """
        try:
            if self._exchange is None:
                logger.error("Exchange not initialized")
                return 0.0

            balance = await self._exchange.fetch_balance()
            # Get USDT balance for futures
            usdt_balance = balance.get("USDT", {})
            total = usdt_balance.get("total", 0.0)
            return float(total) if total else 0.0

        except Exception as e:
            logger.error("Failed to fetch account balance: %s", str(e))
            return 0.0

    async def _set_leverage(self, symbol: str, leverage: int) -> None:
        """Set leverage for a symbol on the exchange.

        Args:
            symbol: Trading pair symbol.
            leverage: Leverage multiplier to set.
        """
        try:
            if self._exchange is None:
                return

            await self._exchange.set_leverage(leverage, symbol)
            logger.info("Leverage set to %sx for %s", leverage, symbol)

        except Exception as e:
            logger.warning(
                "Failed to set leverage for %s: %s (may already be set)",
                symbol, str(e),
            )

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Current price, or None on failure.
        """
        try:
            if self._exchange is None:
                return None

            ticker = await self._exchange.fetch_ticker(symbol)
            return float(ticker.get("last", 0.0))

        except Exception as e:
            logger.error("Failed to get price for %s: %s", symbol, str(e))
            return None

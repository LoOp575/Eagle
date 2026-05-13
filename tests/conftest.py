"""
Shared pytest fixtures for the EAGLE test suite.

Provides reusable sample data: configurations, OHLCV bars, market data,
feature vectors, and trade signals for use across all test modules.
"""

from __future__ import annotations

import sys
import time
from typing import List

import numpy as np

sys.path.insert(0, ".")

from config import (
    AlertConfig,
    ExchangeConfig,
    IndicatorConfig,
    RiskConfig,
    ScanningConfig,
    ScoringConfig,
    SystemConfig,
    TradingConfig,
)
from features.feature_vector import FeatureVector
from scanner.data_models import MarketData, OHLCVBar, TickerData
from scoring.signal_models import Confidence, SignalType, TradeSignal


def pytest_configure(config):
    """Ensure project root is on sys.path for imports."""
    import os
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)


def sample_config() -> SystemConfig:
    """Create a test SystemConfig with sensible defaults.

    Returns:
        SystemConfig suitable for testing.
    """
    return SystemConfig(
        exchange=ExchangeConfig(
            exchange="binance",
            binance_api_key="test_key",
            binance_api_secret="test_secret",
            sandbox_mode=True,
            rate_limit_ms=50,
            max_concurrent_requests=5,
        ),
        scanning=ScanningConfig(
            scan_interval_seconds=30,
            default_timeframe="5m",
            ohlcv_limit=100,
            order_book_depth=20,
            min_volume_usd=1_000_000.0,
            max_symbols=50,
        ),
        indicators=IndicatorConfig(
            volume_period=20,
            oi_period=10,
            funding_period=20,
            compression_period=20,
            atr_period=14,
            ema_fast=8,
            ema_slow=21,
        ),
        scoring=ScoringConfig(
            volume_z_weight=0.25,
            oi_change_weight=0.20,
            oi_acceleration_weight=0.10,
            compression_weight=0.15,
            funding_weight=0.10,
            momentum_weight=0.10,
            mcap_weight=0.10,
        ),
        risk=RiskConfig(
            max_risk_per_trade=0.01,
            max_positions=3,
            daily_max_loss=0.05,
            consecutive_loss_limit=3,
            max_leverage=10,
            position_size_method="kelly",
            cooldown_after_loss_seconds=300,
        ),
        alerts=AlertConfig(
            telegram_enabled=False,
            telegram_bot_token="",
            telegram_chat_id="",
            console_enabled=True,
            json_feed_enabled=True,
        ),
        trading=TradingConfig(
            trading_enabled=False,
            order_type="limit",
            slippage_tolerance=0.001,
            stop_loss_atr_multiplier=1.5,
            take_profit_atr_multiplier=3.0,
        ),
    )


def sample_ohlcv_data() -> List[OHLCVBar]:
    """Generate 100 bars of realistic synthetic OHLCV data.

    Uses a random walk with drift to simulate price action with
    realistic volume patterns.

    Returns:
        List of 100 OHLCVBar instances.
    """
    np.random.seed(42)
    n_bars = 100
    base_price = 50000.0
    base_volume = 1000.0

    # Random walk with slight upward drift
    returns = np.random.normal(0.0005, 0.015, n_bars)
    prices = base_price * np.cumprod(1.0 + returns)

    bars: List[OHLCVBar] = []
    start_ts = int(time.time() * 1000) - n_bars * 300_000  # 5m bars

    for i in range(n_bars):
        close = prices[i]
        # Generate open, high, low relative to close
        intra_volatility = abs(np.random.normal(0, 0.005))
        open_price = close * (1.0 + np.random.normal(0, 0.003))
        high = max(close, open_price) * (1.0 + intra_volatility)
        low = min(close, open_price) * (1.0 - intra_volatility)
        volume = base_volume * (1.0 + abs(np.random.normal(0, 0.5)))

        bars.append(OHLCVBar(
            timestamp=start_ts + i * 300_000,
            open=round(open_price, 2),
            high=round(high, 2),
            low=round(low, 2),
            close=round(close, 2),
            volume=round(volume, 4),
        ))

    return bars


def sample_market_data() -> MarketData:
    """Create a complete MarketData instance for one symbol.

    Returns:
        MarketData with OHLCV, OI, funding rate, and ticker data.
    """
    bars = sample_ohlcv_data()
    last_price = bars[-1].close

    return MarketData(
        symbol="BTC/USDT:USDT",
        ohlcv=bars,
        open_interest=150_000_000.0,
        open_interest_history=[
            {"oi": 145_000_000.0, "timestamp": bars[-3].timestamp},
            {"oi": 147_000_000.0, "timestamp": bars[-2].timestamp},
            {"oi": 150_000_000.0, "timestamp": bars[-1].timestamp},
        ],
        funding_rate=0.0001,
        funding_rate_history=[
            {"rate": 0.00008},
            {"rate": 0.00009},
            {"rate": 0.0001},
            {"rate": 0.00012},
            {"rate": 0.0001},
        ],
        ticker=TickerData(
            symbol="BTC/USDT:USDT",
            last_price=last_price,
            bid=last_price - 1.0,
            ask=last_price + 1.0,
            high_24h=last_price * 1.03,
            low_24h=last_price * 0.97,
            volume_24h=5_000_000.0,
            quote_volume_24h=250_000_000_000.0,
            change_24h=last_price * 0.02,
            change_pct_24h=2.0,
        ),
        order_book=None,
        timestamp=int(time.time() * 1000),
    )


def sample_feature_vector() -> FeatureVector:
    """Create a pre-computed FeatureVector with realistic values.

    Represents a moderately bullish setup with volume and OI support.

    Returns:
        FeatureVector with bullish-leaning values.
    """
    return FeatureVector(
        symbol="BTC/USDT:USDT",
        timestamp=int(time.time() * 1000),
        volume_z=2.5,
        volume_acceleration=0.8,
        volume_anomaly=True,
        oi_change=0.08,
        oi_acceleration=0.04,
        oi_regime="ACCUMULATION",
        funding_pressure=-0.15,
        crowded_trade_index=0.002,
        squeeze_potential=0.6,
        price_compression=0.15,
        atr_ratio=0.025,
        breakout_energy=0.7,
        is_ignition=True,
        mcap_factor=0.4,
        orderbook_depth=0.6,
        liquidity_thinness=0.3,
        trend_strength=0.7,
        momentum_acceleration=0.005,
        mtf_alignment=0.85,
        volume_rank=0.9,
        breakout_rank=0.8,
        attention_score=0.86,
    )


def sample_trade_signal() -> TradeSignal:
    """Create a sample HIGH_PROBABILITY_LONG TradeSignal.

    Returns:
        TradeSignal representing a strong long setup.
    """
    return TradeSignal(
        symbol="BTC/USDT:USDT",
        signal_type=SignalType.HIGH_PROBABILITY_LONG,
        pump_score=82.5,
        dump_score=18.3,
        confidence=Confidence.HIGH,
        reasons=[
            "Volume spike anomaly detected (z=2.5)",
            "OI accumulation increasing (+8.0%)",
            "Negative funding indicating squeeze potential (pressure=-0.150)",
            "Strong upward momentum (strength=0.70)",
        ],
        timestamp=time.time(),
        entry_price=50000.0,
        suggested_stop_loss=-0.0375,
        suggested_take_profit=0.075,
    )

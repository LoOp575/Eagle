"""
Configuration module for the Crypto Futures Intelligence System.

Uses dataclasses with environment variable loading via os.environ.
All system parameters are defined here with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def _env(key: str, default: str = "") -> str:
    """Get environment variable with default."""
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    val = os.environ.get(key, str(default)).lower()
    return val in ("true", "1", "yes")


def _env_int(key: str, default: int = 0) -> int:
    """Get integer environment variable."""
    try:
        return int(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


def _env_float(key: str, default: float = 0.0) -> float:
    """Get float environment variable."""
    try:
        return float(os.environ.get(key, str(default)))
    except (ValueError, TypeError):
        return default


@dataclass
class ExchangeConfig:
    """Exchange connection settings."""

    exchange: str = field(default_factory=lambda: _env("EXCHANGE", "binance"))
    binance_api_key: str = field(default_factory=lambda: _env("BINANCE_API_KEY"))
    binance_api_secret: str = field(default_factory=lambda: _env("BINANCE_API_SECRET"))
    bybit_api_key: str = field(default_factory=lambda: _env("BYBIT_API_KEY"))
    bybit_api_secret: str = field(default_factory=lambda: _env("BYBIT_API_SECRET"))
    sandbox_mode: bool = False
    rate_limit_ms: int = 50
    max_concurrent_requests: int = 10


@dataclass
class ScanningConfig:
    """Market scanning parameters."""

    scan_interval_seconds: int = field(
        default_factory=lambda: _env_int("SCAN_INTERVAL_SECONDS", 60)
    )
    default_timeframe: str = "5m"
    ohlcv_limit: int = 100
    order_book_depth: int = 20
    min_volume_usd: float = 1_000_000.0
    max_symbols: int = 100


@dataclass
class IndicatorConfig:
    """Indicator calculation periods."""

    volume_period: int = 20
    oi_period: int = 10
    funding_period: int = 20
    compression_period: int = 20
    atr_period: int = 14
    ema_fast: int = 8
    ema_slow: int = 21


@dataclass
class ScoringConfig:
    """Scoring weights for pump/dump probability calculation."""

    volume_z_weight: float = 0.25
    oi_change_weight: float = 0.20
    oi_acceleration_weight: float = 0.10
    compression_weight: float = 0.15
    funding_weight: float = 0.10
    momentum_weight: float = 0.10
    mcap_weight: float = 0.10

    # Thresholds
    pump_threshold: float = 0.70
    dump_threshold: float = 0.70
    alert_threshold: float = 0.60


@dataclass
class RiskConfig:
    """Risk management parameters."""

    max_risk_per_trade: float = field(
        default_factory=lambda: _env_float("RISK_PER_TRADE", 0.01)
    )
    max_positions: int = field(
        default_factory=lambda: _env_int("MAX_POSITIONS", 3)
    )
    daily_max_loss: float = field(
        default_factory=lambda: _env_float("DAILY_MAX_LOSS", 0.05)
    )
    consecutive_loss_limit: int = 3
    max_leverage: int = 10
    position_size_method: str = "kelly"
    cooldown_after_loss_seconds: int = 300


@dataclass
class AlertConfig:
    """Alert and notification settings."""

    telegram_enabled: bool = field(
        default_factory=lambda: _env_bool("TELEGRAM_ENABLED", False)
    )
    telegram_bot_token: str = field(
        default_factory=lambda: _env("TELEGRAM_BOT_TOKEN")
    )
    telegram_chat_id: str = field(
        default_factory=lambda: _env("TELEGRAM_CHAT_ID")
    )
    console_enabled: bool = True
    json_feed_enabled: bool = True
    json_feed_path: str = "output/signals.json"
    alert_cooldown_seconds: int = 300


@dataclass
class TradingConfig:
    """Trading execution settings."""

    trading_enabled: bool = field(
        default_factory=lambda: _env_bool("TRADING_ENABLED", False)
    )
    order_type: str = "limit"
    slippage_tolerance: float = 0.001
    stop_loss_atr_multiplier: float = 1.5
    take_profit_atr_multiplier: float = 3.0
    trailing_stop_enabled: bool = True
    trailing_stop_activation: float = 0.02
    trailing_stop_callback: float = 0.01


@dataclass
class SystemConfig:
    """Root configuration combining all sub-configs."""

    exchange: ExchangeConfig = field(default_factory=ExchangeConfig)
    scanning: ScanningConfig = field(default_factory=ScanningConfig)
    indicators: IndicatorConfig = field(default_factory=IndicatorConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    alerts: AlertConfig = field(default_factory=AlertConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/system.log"


def load_config() -> SystemConfig:
    """Load system configuration from environment variables and defaults.

    Call this at application startup. It reads from os.environ which
    should be populated from .env file via python-dotenv.

    Returns:
        SystemConfig: Fully populated configuration object.
    """
    return SystemConfig()

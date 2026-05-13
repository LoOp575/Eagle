"""
Data models for the market scanner module.

Defines dataclasses representing all market data structures used throughout
the system for OHLCV data, ticker info, order books, and scan results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class OHLCVBar:
    """Single OHLCV candlestick bar."""

    timestamp: int  # Unix timestamp in milliseconds
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def is_bullish(self) -> bool:
        """Check if the bar closed higher than it opened."""
        return self.close >= self.open

    @property
    def body_size(self) -> float:
        """Absolute size of the candle body."""
        return abs(self.close - self.open)

    @property
    def range_size(self) -> float:
        """Total high-low range of the bar."""
        return self.high - self.low

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "timestamp": self.timestamp,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }

    @classmethod
    def from_list(cls, data: List[Any]) -> "OHLCVBar":
        """Create OHLCVBar from CCXT-style list [timestamp, o, h, l, c, v]."""
        return cls(
            timestamp=int(data[0]),
            open=float(data[1]),
            high=float(data[2]),
            low=float(data[3]),
            close=float(data[4]),
            volume=float(data[5]),
        )


@dataclass
class TickerData:
    """Ticker/market summary data for a symbol."""

    symbol: str
    last_price: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    high_24h: Optional[float] = None
    low_24h: Optional[float] = None
    volume_24h: Optional[float] = None
    quote_volume_24h: Optional[float] = None
    change_24h: Optional[float] = None
    change_pct_24h: Optional[float] = None
    timestamp: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "symbol": self.symbol,
            "last_price": self.last_price,
            "bid": self.bid,
            "ask": self.ask,
            "high_24h": self.high_24h,
            "low_24h": self.low_24h,
            "volume_24h": self.volume_24h,
            "quote_volume_24h": self.quote_volume_24h,
            "change_24h": self.change_24h,
            "change_pct_24h": self.change_pct_24h,
            "timestamp": self.timestamp,
        }


@dataclass
class OrderBookLevel:
    """Single level in the order book."""

    price: float
    quantity: float


@dataclass
class OrderBookSnapshot:
    """Order book snapshot for a symbol."""

    symbol: str
    timestamp: int
    bids: List[OrderBookLevel] = field(default_factory=list)
    asks: List[OrderBookLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        """Best (highest) bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Best (lowest) ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def spread_pct(self) -> Optional[float]:
        """Bid-ask spread as percentage of mid price."""
        if self.best_bid is not None and self.best_ask is not None:
            mid = (self.best_bid + self.best_ask) / 2
            if mid > 0:
                return (self.best_ask - self.best_bid) / mid
        return None

    @property
    def bid_depth(self) -> float:
        """Total bid-side liquidity (sum of quantities)."""
        return sum(level.quantity for level in self.bids)

    @property
    def ask_depth(self) -> float:
        """Total ask-side liquidity (sum of quantities)."""
        return sum(level.quantity for level in self.asks)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "bids": [[b.price, b.quantity] for b in self.bids],
            "asks": [[a.price, a.quantity] for a in self.asks],
        }


@dataclass
class MarketData:
    """Complete market data snapshot for a single symbol.

    Contains all data needed for indicator calculations:
    OHLCV history, open interest, funding rates, ticker, and order book.
    """

    symbol: str
    ohlcv: List[OHLCVBar] = field(default_factory=list)
    open_interest: Optional[float] = None
    open_interest_history: List[Dict[str, Any]] = field(default_factory=list)
    funding_rate: Optional[float] = None
    funding_rate_history: List[Dict[str, Any]] = field(default_factory=list)
    ticker: Optional[TickerData] = None
    order_book: Optional[OrderBookSnapshot] = None
    timestamp: Optional[int] = None

    @property
    def has_ohlcv(self) -> bool:
        """Check if OHLCV data is available."""
        return len(self.ohlcv) > 0

    @property
    def last_close(self) -> Optional[float]:
        """Get the most recent closing price."""
        if self.ohlcv:
            return self.ohlcv[-1].close
        return None

    @property
    def closes(self) -> List[float]:
        """Get list of closing prices."""
        return [bar.close for bar in self.ohlcv]

    @property
    def volumes(self) -> List[float]:
        """Get list of volumes."""
        return [bar.volume for bar in self.ohlcv]

    @property
    def highs(self) -> List[float]:
        """Get list of high prices."""
        return [bar.high for bar in self.ohlcv]

    @property
    def lows(self) -> List[float]:
        """Get list of low prices."""
        return [bar.low for bar in self.ohlcv]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "symbol": self.symbol,
            "ohlcv_count": len(self.ohlcv),
            "open_interest": self.open_interest,
            "funding_rate": self.funding_rate,
            "ticker": self.ticker.to_dict() if self.ticker else None,
            "timestamp": self.timestamp,
        }


@dataclass
class ScanResult:
    """Result of scanning a single symbol.

    Encapsulates the market data along with metadata about the scan.
    """

    symbol: str
    market_data: MarketData
    timestamp: int
    success: bool = True
    error: Optional[str] = None
    scan_duration_ms: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "symbol": self.symbol,
            "success": self.success,
            "error": self.error,
            "timestamp": self.timestamp,
            "scan_duration_ms": self.scan_duration_ms,
            "market_data": self.market_data.to_dict() if self.success else None,
        }

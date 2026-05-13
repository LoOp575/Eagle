"""
Market Scanner module for fetching futures market data.

Provides async methods for fetching OHLCV, open interest, funding rates,
ticker data, and order books from cryptocurrency exchanges via CCXT.
Includes rate limiting, exponential backoff retry, and per-symbol error handling.
"""

from __future__ import annotations

import asyncio
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, TypeVar, Coroutine

from scanner.data_models import (
    MarketData,
    OHLCVBar,
    OrderBookLevel,
    OrderBookSnapshot,
    ScanResult,
    TickerData,
)

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Coroutine[Any, Any, Any]])


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exceptions: tuple = (Exception,),
) -> Callable[[F], F]:
    """Decorator for async functions that retries with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay between retries in seconds.
        max_delay: Maximum delay between retries in seconds.
        exceptions: Tuple of exception types to catch and retry on.

    Returns:
        Decorated function with retry logic.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception: Optional[Exception] = None
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries:
                        delay = min(base_delay * (2 ** attempt), max_delay)
                        logger.warning(
                            "Attempt %d/%d failed for %s: %s. "
                            "Retrying in %.1fs...",
                            attempt + 1,
                            max_retries + 1,
                            func.__name__,
                            str(e),
                            delay,
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "All %d attempts failed for %s: %s",
                            max_retries + 1,
                            func.__name__,
                            str(e),
                        )
            raise last_exception  # type: ignore[misc]

        return wrapper  # type: ignore[return-value]

    return decorator


class MarketScanner:
    """Async market data scanner using CCXT for exchange connectivity.

    Fetches futures market data including OHLCV, open interest, funding rates,
    tickers, and order books. Implements rate limiting and error isolation
    so a single symbol failure does not crash the entire scan.

    Args:
        config: SystemConfig instance with exchange and scanning parameters.
    """

    def __init__(self, config: Any) -> None:
        """Initialize the MarketScanner.

        Args:
            config: SystemConfig object containing exchange and scanning settings.
        """
        self._config = config
        self._exchange: Any = None
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._symbols: List[str] = []
        self._initialized: bool = False

    async def initialize(self) -> None:
        """Initialize exchange connection and rate limiter.

        Creates the CCXT pro exchange instance and sets up the semaphore
        for concurrent request limiting.
        """
        import ccxt.pro as ccxtpro  # type: ignore[import]

        exchange_id = self._config.exchange.exchange.lower()
        exchange_class = getattr(ccxtpro, exchange_id, None)

        if exchange_class is None:
            raise ValueError(f"Unsupported exchange: {exchange_id}")

        # Determine API credentials based on exchange
        api_key = ""
        api_secret = ""
        if exchange_id == "binance":
            api_key = self._config.exchange.binance_api_key
            api_secret = self._config.exchange.binance_api_secret
        elif exchange_id == "bybit":
            api_key = self._config.exchange.bybit_api_key
            api_secret = self._config.exchange.bybit_api_secret

        exchange_options: Dict[str, Any] = {
            "apiKey": api_key,
            "secret": api_secret,
            "sandbox": self._config.exchange.sandbox_mode,
            "enableRateLimit": True,
            "options": {
                "defaultType": "swap",
                "adjustForTimeDifference": True,
            },
        }

        self._exchange = exchange_class(exchange_options)
        self._semaphore = asyncio.Semaphore(
            self._config.exchange.max_concurrent_requests
        )
        self._initialized = True
        logger.info("MarketScanner initialized with exchange: %s", exchange_id)

    async def close(self) -> None:
        """Close the exchange connection and clean up resources."""
        if self._exchange is not None:
            await self._exchange.close()
            self._exchange = None
        self._initialized = False
        logger.info("MarketScanner closed")

    def _ensure_initialized(self) -> None:
        """Raise if the scanner has not been initialized."""
        if not self._initialized or self._exchange is None:
            raise RuntimeError(
                "MarketScanner not initialized. Call initialize() first."
            )

    async def fetch_all_futures_symbols(self) -> List[str]:
        """Fetch all available futures/swap trading symbols.

        Loads exchange markets and filters for active swap/futures contracts.

        Returns:
            List of symbol strings (e.g., ['BTC/USDT:USDT', 'ETH/USDT:USDT']).
        """
        self._ensure_initialized()

        async with self._semaphore:  # type: ignore[union-attr]
            markets = await self._exchange.load_markets()

        symbols = []
        for symbol, market in markets.items():
            if (
                market.get("swap", False)
                and market.get("active", True)
                and market.get("quote", "") == "USDT"
                and market.get("linear", False)
            ):
                symbols.append(symbol)

        # Filter by minimum volume if ticker data available
        self._symbols = sorted(symbols)
        logger.info("Found %d futures symbols", len(self._symbols))
        return self._symbols

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 100,
    ) -> List[OHLCVBar]:
        """Fetch OHLCV candlestick data for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT:USDT').
            timeframe: Candlestick timeframe (e.g., '1m', '5m', '1h').
            limit: Number of bars to fetch.

        Returns:
            List of OHLCVBar dataclass instances.
        """
        self._ensure_initialized()

        async with self._semaphore:  # type: ignore[union-attr]
            raw_data = await self._exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, limit=limit
            )

        bars = [OHLCVBar.from_list(candle) for candle in raw_data]
        return bars

    async def fetch_ohlcv_batch(
        self,
        symbols: List[str],
        timeframe: str = "5m",
        limit: int = 100,
    ) -> Dict[str, List[OHLCVBar]]:
        """Fetch OHLCV data for multiple symbols concurrently.

        Args:
            symbols: List of trading pair symbols.
            timeframe: Candlestick timeframe.
            limit: Number of bars to fetch per symbol.

        Returns:
            Dict mapping symbol to list of OHLCVBar instances.
        """
        results: Dict[str, List[OHLCVBar]] = {}

        async def _fetch_one(sym: str) -> None:
            try:
                bars = await self.fetch_ohlcv(sym, timeframe, limit)
                results[sym] = bars
            except Exception as e:
                logger.error("Failed to fetch OHLCV for %s: %s", sym, str(e))
                results[sym] = []

        tasks = [_fetch_one(sym) for sym in symbols]
        await asyncio.gather(*tasks)
        return results

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def fetch_open_interest(self, symbol: str) -> Optional[float]:
        """Fetch current open interest for a symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Open interest value in base currency, or None if unavailable.
        """
        self._ensure_initialized()

        try:
            async with self._semaphore:  # type: ignore[union-attr]
                oi_data = await self._exchange.fetch_open_interest(symbol)
            if oi_data and "openInterestAmount" in oi_data:
                return float(oi_data["openInterestAmount"])
            elif oi_data and "openInterest" in oi_data:
                return float(oi_data["openInterest"])
        except Exception as e:
            logger.warning(
                "Open interest not available for %s: %s", symbol, str(e)
            )
        return None

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        """Fetch current funding rate for a perpetual futures symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            Current funding rate as a decimal, or None if unavailable.
        """
        self._ensure_initialized()

        try:
            async with self._semaphore:  # type: ignore[union-attr]
                funding_data = await self._exchange.fetch_funding_rate(symbol)
            if funding_data and "fundingRate" in funding_data:
                return float(funding_data["fundingRate"])
        except Exception as e:
            logger.warning(
                "Funding rate not available for %s: %s", symbol, str(e)
            )
        return None

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def fetch_ticker(self, symbol: str) -> Optional[TickerData]:
        """Fetch current ticker data for a symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            TickerData instance, or None if fetch fails.
        """
        self._ensure_initialized()

        async with self._semaphore:  # type: ignore[union-attr]
            raw = await self._exchange.fetch_ticker(symbol)

        if not raw:
            return None

        return TickerData(
            symbol=symbol,
            last_price=float(raw.get("last", 0) or 0),
            bid=float(raw["bid"]) if raw.get("bid") else None,
            ask=float(raw["ask"]) if raw.get("ask") else None,
            high_24h=float(raw["high"]) if raw.get("high") else None,
            low_24h=float(raw["low"]) if raw.get("low") else None,
            volume_24h=float(raw["baseVolume"]) if raw.get("baseVolume") else None,
            quote_volume_24h=(
                float(raw["quoteVolume"]) if raw.get("quoteVolume") else None
            ),
            change_24h=float(raw["change"]) if raw.get("change") else None,
            change_pct_24h=(
                float(raw["percentage"]) if raw.get("percentage") else None
            ),
            timestamp=int(raw["timestamp"]) if raw.get("timestamp") else None,
        )

    async def fetch_ticker_batch(
        self, symbols: List[str]
    ) -> Dict[str, Optional[TickerData]]:
        """Fetch ticker data for multiple symbols concurrently.

        Args:
            symbols: List of trading pair symbols.

        Returns:
            Dict mapping symbol to TickerData (or None on failure).
        """
        results: Dict[str, Optional[TickerData]] = {}

        async def _fetch_one(sym: str) -> None:
            try:
                ticker = await self.fetch_ticker(sym)
                results[sym] = ticker
            except Exception as e:
                logger.error("Failed to fetch ticker for %s: %s", sym, str(e))
                results[sym] = None

        tasks = [_fetch_one(sym) for sym in symbols]
        await asyncio.gather(*tasks)
        return results

    @retry_with_backoff(max_retries=3, base_delay=1.0)
    async def fetch_order_book(
        self, symbol: str, limit: int = 20
    ) -> Optional[OrderBookSnapshot]:
        """Fetch order book snapshot for a symbol.

        Args:
            symbol: Trading pair symbol.
            limit: Number of levels per side.

        Returns:
            OrderBookSnapshot instance, or None if fetch fails.
        """
        self._ensure_initialized()

        async with self._semaphore:  # type: ignore[union-attr]
            raw = await self._exchange.fetch_order_book(symbol, limit=limit)

        if not raw:
            return None

        bids = [
            OrderBookLevel(price=float(level[0]), quantity=float(level[1]))
            for level in raw.get("bids", [])
        ]
        asks = [
            OrderBookLevel(price=float(level[0]), quantity=float(level[1]))
            for level in raw.get("asks", [])
        ]

        return OrderBookSnapshot(
            symbol=symbol,
            timestamp=int(raw.get("timestamp", 0) or int(time.time() * 1000)),
            bids=bids,
            asks=asks,
        )

    async def scan_symbol(self, symbol: str) -> ScanResult:
        """Perform a complete data scan for a single symbol.

        Fetches OHLCV, open interest, funding rate, ticker, and order book.
        Handles errors gracefully - partial data is still returned.

        Args:
            symbol: Trading pair symbol.

        Returns:
            ScanResult containing all available market data.
        """
        start_time = time.time()
        timestamp_ms = int(start_time * 1000)

        try:
            # Fetch all data concurrently for this symbol
            ohlcv_task = self.fetch_ohlcv(
                symbol,
                self._config.scanning.default_timeframe,
                self._config.scanning.ohlcv_limit,
            )
            oi_task = self.fetch_open_interest(symbol)
            funding_task = self.fetch_funding_rate(symbol)
            ticker_task = self.fetch_ticker(symbol)
            order_book_task = self.fetch_order_book(
                symbol, self._config.scanning.order_book_depth
            )

            results = await asyncio.gather(
                ohlcv_task,
                oi_task,
                funding_task,
                ticker_task,
                order_book_task,
                return_exceptions=True,
            )

            # Extract results, handling individual failures
            ohlcv = results[0] if not isinstance(results[0], Exception) else []
            open_interest = (
                results[1] if not isinstance(results[1], Exception) else None
            )
            funding_rate = (
                results[2] if not isinstance(results[2], Exception) else None
            )
            ticker = (
                results[3] if not isinstance(results[3], Exception) else None
            )
            order_book = (
                results[4] if not isinstance(results[4], Exception) else None
            )

            # Log any partial failures
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    labels = [
                        "OHLCV", "OI", "funding", "ticker", "orderbook"
                    ]
                    logger.warning(
                        "Partial failure for %s [%s]: %s",
                        symbol,
                        labels[i],
                        str(result),
                    )

            market_data = MarketData(
                symbol=symbol,
                ohlcv=ohlcv if isinstance(ohlcv, list) else [],
                open_interest=open_interest,
                funding_rate=funding_rate,
                ticker=ticker,
                order_book=order_book,
                timestamp=timestamp_ms,
            )

            elapsed_ms = (time.time() - start_time) * 1000

            return ScanResult(
                symbol=symbol,
                market_data=market_data,
                timestamp=timestamp_ms,
                success=True,
                scan_duration_ms=elapsed_ms,
            )

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            logger.error("Complete scan failure for %s: %s", symbol, str(e))
            return ScanResult(
                symbol=symbol,
                market_data=MarketData(symbol=symbol),
                timestamp=timestamp_ms,
                success=False,
                error=str(e),
                scan_duration_ms=elapsed_ms,
            )

    async def scan_all(self) -> List[ScanResult]:
        """Scan all futures symbols and return results.

        Fetches symbols if not already loaded, then scans each one.
        Results are returned for all symbols, including failures.

        Returns:
            List of ScanResult instances for all scanned symbols.
        """
        self._ensure_initialized()

        if not self._symbols:
            await self.fetch_all_futures_symbols()

        logger.info("Starting full scan of %d symbols", len(self._symbols))
        start_time = time.time()

        # Limit concurrent symbol scans
        results: List[ScanResult] = []
        batch_size = self._config.exchange.max_concurrent_requests

        for i in range(0, len(self._symbols), batch_size):
            batch = self._symbols[i : i + batch_size]
            batch_results = await asyncio.gather(
                *[self.scan_symbol(sym) for sym in batch]
            )
            results.extend(batch_results)

        elapsed = time.time() - start_time
        successful = sum(1 for r in results if r.success)
        logger.info(
            "Full scan complete: %d/%d symbols successful in %.1fs",
            successful,
            len(results),
            elapsed,
        )

        return results

    async def __aenter__(self) -> "MarketScanner":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Async context manager exit."""
        await self.close()

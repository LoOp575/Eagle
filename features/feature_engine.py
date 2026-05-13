"""
Feature engine - orchestrates all indicator computations.

The FeatureEngine is the central coordinator that takes raw MarketData
and produces a complete FeatureVector by calling each indicator analyzer
in sequence. It handles missing data gracefully, providing safe defaults
when specific data points are unavailable.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from config import IndicatorConfig
from features.feature_vector import FeatureVector
from indicators.attention import AttentionAnalyzer
from indicators.compression import CompressionAnalyzer
from indicators.funding import FundingAnalyzer
from indicators.liquidity import LiquidityAnalyzer
from indicators.momentum import MomentumAnalyzer
from indicators.open_interest import OIAnalyzer
from indicators.volume import VolumeAnalyzer
from scanner.data_models import MarketData


class FeatureEngine:
    """Orchestrates all indicator computations to produce feature vectors.

    Initializes all analyzer instances and coordinates their execution
    to transform raw MarketData into a complete FeatureVector ready for
    scoring.
    """

    def __init__(self, config: IndicatorConfig) -> None:
        """Initialize the FeatureEngine with indicator configuration.

        Args:
            config: IndicatorConfig containing periods and parameters
                    for all indicator calculations.
        """
        self.config = config

        # Initialize all analyzers
        self.volume_analyzer = VolumeAnalyzer()
        self.oi_analyzer = OIAnalyzer()
        self.funding_analyzer = FundingAnalyzer()
        self.compression_analyzer = CompressionAnalyzer()
        self.liquidity_analyzer = LiquidityAnalyzer()
        self.momentum_analyzer = MomentumAnalyzer()
        self.attention_analyzer = AttentionAnalyzer()

    def compute_features(
        self,
        market_data: MarketData,
        all_market_data: Optional[List[MarketData]] = None,
    ) -> FeatureVector:
        """Compute all features for a single symbol's market data.

        Orchestrates all indicator computations and assembles the results
        into a FeatureVector. Handles missing data gracefully by using
        safe defaults (0.0 or neutral values).

        Args:
            market_data: Complete market data snapshot for one symbol.
            all_market_data: Optional list of market data for all monitored
                             symbols, needed for ranking/attention calculations.

        Returns:
            FeatureVector with all computed indicator values.
        """
        fv = FeatureVector(
            symbol=market_data.symbol,
            timestamp=market_data.timestamp or int(time.time() * 1000),
        )

        # Extract arrays from OHLCV data
        volumes = np.asarray(market_data.volumes, dtype=np.float64) if market_data.has_ohlcv else np.array([])
        closes = np.asarray(market_data.closes, dtype=np.float64) if market_data.has_ohlcv else np.array([])
        highs = np.asarray(market_data.highs, dtype=np.float64) if market_data.has_ohlcv else np.array([])
        lows = np.asarray(market_data.lows, dtype=np.float64) if market_data.has_ohlcv else np.array([])

        current_close = market_data.last_close or 0.0

        # --- Volume indicators ---
        fv.volume_z = self.volume_analyzer.compute_volume_z(
            volumes, period=self.config.volume_period
        )
        # For acceleration, compute z-score of previous bar
        if len(volumes) > self.config.volume_period:
            prev_z = self.volume_analyzer.compute_volume_z(
                volumes[:-1], period=self.config.volume_period
            )
        else:
            prev_z = 0.0
        fv.volume_acceleration = self.volume_analyzer.compute_volume_acceleration(
            fv.volume_z, prev_z
        )
        anomaly_result = self.volume_analyzer.detect_anomaly(fv.volume_z)
        fv.volume_anomaly = anomaly_result[0]

        # --- Open interest indicators ---
        fv.oi_change = self._compute_oi_change(market_data)
        fv.oi_acceleration = self._compute_oi_acceleration(market_data)
        fv.oi_regime = self._compute_oi_regime(market_data, closes)

        # --- Funding indicators ---
        fv.funding_pressure = self._compute_funding_pressure(market_data)
        fv.crowded_trade_index = self.funding_analyzer.compute_crowded_trade_index(
            market_data.funding_rate or 0.0, fv.oi_change
        )
        fv.squeeze_potential = self.funding_analyzer.detect_squeeze_potential(
            fv.funding_pressure, fv.oi_change
        )

        # --- Compression indicators ---
        if current_close > 0 and len(highs) > 0:
            fv.price_compression = self.compression_analyzer.compute_price_compression(
                highs, lows, current_close, period=self.config.compression_period
            )
            fv.atr_ratio = self.compression_analyzer.compute_atr_ratio(
                highs, lows, closes, period=self.config.atr_period
            )
            # For breakout energy, use compression history (simplified: just current value)
            compression_history = [fv.price_compression]
            energy_result = self.compression_analyzer.detect_breakout_energy(
                compression_history, fv.atr_ratio
            )
            fv.breakout_energy = energy_result[0]
            fv.is_ignition = energy_result[1]

        # --- Liquidity indicators ---
        fv.mcap_factor = self._compute_mcap_factor(market_data)
        fv.orderbook_depth = self._compute_orderbook_depth(market_data)
        volume_24h = self._get_volume_24h(market_data)
        fv.liquidity_thinness = self.liquidity_analyzer.compute_liquidity_thinness(
            fv.orderbook_depth, volume_24h
        )

        # --- Momentum indicators ---
        if len(closes) >= self.config.ema_slow:
            fv.trend_strength = self.momentum_analyzer.compute_trend_strength(
                closes,
                fast_period=self.config.ema_fast,
                slow_period=self.config.ema_slow,
            )
        fv.momentum_acceleration = self.momentum_analyzer.compute_momentum_acceleration(
            closes, period=self.config.atr_period
        )
        # MTF alignment defaults to 0.5 (single timeframe only)
        # Full MTF calculation requires multi-timeframe data
        fv.mtf_alignment = 0.5

        # --- Attention/ranking indicators ---
        if all_market_data is not None and len(all_market_data) > 0:
            fv.volume_rank, fv.breakout_rank, fv.attention_score = (
                self._compute_attention(market_data, all_market_data)
            )

        return fv

    def _compute_oi_change(self, market_data: MarketData) -> float:
        """Compute OI change from market data."""
        if market_data.open_interest is None:
            return 0.0

        # Try to get previous OI from history
        if market_data.open_interest_history and len(market_data.open_interest_history) >= 2:
            prev_entry = market_data.open_interest_history[-2]
            prev_oi = prev_entry.get("oi", prev_entry.get("open_interest", 0.0))
            if prev_oi and prev_oi > 0:
                return self.oi_analyzer.compute_oi_change(
                    market_data.open_interest, prev_oi
                )

        return 0.0

    def _compute_oi_acceleration(self, market_data: MarketData) -> float:
        """Compute OI acceleration from history."""
        if not market_data.open_interest_history or len(market_data.open_interest_history) < 3:
            return 0.0

        # Compute sequential OI changes
        oi_changes = []
        for i in range(1, len(market_data.open_interest_history)):
            curr = market_data.open_interest_history[i]
            prev = market_data.open_interest_history[i - 1]
            curr_oi = curr.get("oi", curr.get("open_interest", 0.0))
            prev_oi = prev.get("oi", prev.get("open_interest", 0.0))
            if prev_oi and prev_oi > 0:
                oi_changes.append(self.oi_analyzer.compute_oi_change(curr_oi, prev_oi))

        return self.oi_analyzer.compute_oi_acceleration(oi_changes)

    def _compute_oi_regime(self, market_data: MarketData, closes: np.ndarray) -> str:
        """Classify OI regime from market data."""
        oi_change = self._compute_oi_change(market_data)

        # Compute price change
        if len(closes) >= 2:
            price_change = (closes[-1] - closes[-2]) / closes[-2] if closes[-2] != 0 else 0.0
        else:
            price_change = 0.0

        return self.oi_analyzer.classify_oi_regime(oi_change, price_change)

    def _compute_funding_pressure(self, market_data: MarketData) -> float:
        """Compute funding pressure from rate history."""
        if not market_data.funding_rate_history and market_data.funding_rate is None:
            return 0.0

        if market_data.funding_rate_history:
            rates = []
            for entry in market_data.funding_rate_history:
                rate = entry.get("rate", entry.get("funding_rate", 0.0))
                if rate is not None:
                    rates.append(rate)
            if rates:
                return self.funding_analyzer.compute_funding_pressure(
                    rates, period=self.config.funding_period
                )

        # Fallback: single rate, no history for comparison
        return 0.0

    def _compute_mcap_factor(self, market_data: MarketData) -> float:
        """Compute market cap manipulation factor."""
        volume_24h = self._get_volume_24h(market_data)
        price = market_data.last_close or 0.0

        if volume_24h <= 0 or price <= 0:
            return 0.0

        return self.liquidity_analyzer.compute_mcap_factor(
            volume_24h=volume_24h,
            price=price,
        )

    def _compute_orderbook_depth(self, market_data: MarketData) -> float:
        """Compute order book depth score."""
        if market_data.order_book is None:
            return 0.5  # Neutral default when no order book data

        return self.liquidity_analyzer.compute_orderbook_depth(
            bids=market_data.order_book.bids,
            asks=market_data.order_book.asks,
        )

    def _get_volume_24h(self, market_data: MarketData) -> float:
        """Extract 24h volume from ticker or OHLCV data."""
        if market_data.ticker and market_data.ticker.volume_24h:
            return market_data.ticker.volume_24h

        # Fallback: sum recent OHLCV volumes (not exact 24h but useful)
        if market_data.has_ohlcv:
            return sum(bar.volume for bar in market_data.ohlcv[-24:])

        return 0.0

    def _compute_attention(
        self, market_data: MarketData, all_market_data: List[MarketData]
    ) -> tuple:
        """Compute attention metrics relative to all symbols.

        Returns:
            Tuple of (volume_rank, breakout_rank, attention_score).
        """
        # Gather all volumes and price changes for ranking
        all_volumes = []
        all_changes = []
        symbol_volume = self._get_volume_24h(market_data)
        symbol_change = 0.0

        if market_data.ticker and market_data.ticker.change_pct_24h is not None:
            symbol_change = market_data.ticker.change_pct_24h

        for md in all_market_data:
            vol = 0.0
            if md.ticker and md.ticker.volume_24h:
                vol = md.ticker.volume_24h
            elif md.has_ohlcv:
                vol = sum(bar.volume for bar in md.ohlcv[-24:])
            all_volumes.append(vol)

            chg = 0.0
            if md.ticker and md.ticker.change_pct_24h is not None:
                chg = md.ticker.change_pct_24h
            all_changes.append(chg)

        volume_rank = self.attention_analyzer.compute_relative_volume_rank(
            symbol_volume, all_volumes
        )
        breakout_rank = self.attention_analyzer.compute_breakout_rank(
            symbol_change, all_changes
        )
        attention_score = self.attention_analyzer.compute_attention_score(
            volume_rank, breakout_rank
        )

        return (volume_rank, breakout_rank, attention_score)

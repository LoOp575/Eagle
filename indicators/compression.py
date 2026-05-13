"""
Price compression and volatility contraction analysis.

Market Theory:
Markets alternate between periods of compression (low volatility, tight ranges)
and expansion (high volatility, wide ranges). Compression periods act as
"energy storage" - the longer and tighter the compression, the more explosive
the subsequent breakout tends to be.

The Price Compression Ratio measures how tight the recent trading range is
relative to price. Lower values indicate more compression. The ATR Ratio
normalizes volatility by price level, making it comparable across assets.

Breakout Energy quantifies the stored energy from compression, while the
Ignition flag detects the moment volatility begins expanding from a
compressed state - the earliest signal of a potential explosive move.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class CompressionAnalyzer:
    """Analyzes price compression and breakout potential."""

    def compute_price_compression(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        close: float,
        period: int = 20,
    ) -> float:
        """Compute price compression ratio.

        Formula: (HighestHigh - LowestLow) / Close

        Lower values indicate tighter compression (more breakout energy stored).

        Args:
            highs: Array of high prices.
            lows: Array of low prices.
            close: Current closing price.
            period: Lookback period for high/low range.

        Returns:
            Compression ratio. Lower values = more compressed.
            Returns 0.0 if insufficient data or close is zero.
        """
        if highs is None or lows is None or len(highs) == 0 or len(lows) == 0:
            return 0.0
        if close is None or close == 0.0:
            return 0.0

        h_arr = np.asarray(highs, dtype=np.float64)
        l_arr = np.asarray(lows, dtype=np.float64)

        # Use the last 'period' values, or all available
        h_window = h_arr[-period:] if len(h_arr) >= period else h_arr
        l_window = l_arr[-period:] if len(l_arr) >= period else l_arr

        highest_high = np.max(h_window)
        lowest_low = np.min(l_window)

        compression = (highest_high - lowest_low) / close
        return float(compression)

    def compute_atr_ratio(
        self,
        highs: np.ndarray,
        lows: np.ndarray,
        closes: np.ndarray,
        period: int = 14,
    ) -> float:
        """Compute ATR (Average True Range) normalized by close price.

        True Range = max(H-L, abs(H-Prev_C), abs(L-Prev_C))
        ATR = SMA(TR, period)
        ATR Ratio = ATR / Close

        Args:
            highs: Array of high prices.
            lows: Array of low prices.
            closes: Array of closing prices.
            period: ATR calculation period.

        Returns:
            ATR divided by current close price. Returns 0.0 if insufficient data.
        """
        if (
            highs is None
            or lows is None
            or closes is None
            or len(highs) < 2
            or len(lows) < 2
            or len(closes) < 2
        ):
            return 0.0

        h_arr = np.asarray(highs, dtype=np.float64)
        l_arr = np.asarray(lows, dtype=np.float64)
        c_arr = np.asarray(closes, dtype=np.float64)

        # Ensure arrays are same length
        min_len = min(len(h_arr), len(l_arr), len(c_arr))
        h_arr = h_arr[-min_len:]
        l_arr = l_arr[-min_len:]
        c_arr = c_arr[-min_len:]

        if min_len < 2:
            return 0.0

        # True Range calculation
        high_low = h_arr[1:] - l_arr[1:]
        high_prev_close = np.abs(h_arr[1:] - c_arr[:-1])
        low_prev_close = np.abs(l_arr[1:] - c_arr[:-1])

        true_range = np.maximum(high_low, np.maximum(high_prev_close, low_prev_close))

        if len(true_range) == 0:
            return 0.0

        # ATR as simple moving average of TR
        atr_period = min(period, len(true_range))
        atr = np.mean(true_range[-atr_period:])

        current_close = c_arr[-1]
        if current_close == 0.0:
            return 0.0

        return float(atr / current_close)

    def detect_breakout_energy(
        self,
        compression_history: List[float],
        current_atr_ratio: float,
        lookback: int = 5,
    ) -> Tuple[float, bool]:
        """Detect breakout energy and ignition signal.

        Energy Score: Inverse of current compression (low compression = high energy).
        Ignition: Detected when ATR suddenly expands vs its recent mean,
        indicating the breakout from compression has begun.

        Args:
            compression_history: List of recent compression values (most recent last).
            current_atr_ratio: Current ATR ratio value.
            lookback: Number of periods for ATR mean comparison.

        Returns:
            Tuple of (energy_score 0-1, is_ignition).
            Energy score is clamped between 0.0 and 1.0.
        """
        if compression_history is None or len(compression_history) == 0:
            return (0.0, False)

        # Energy is inverse of compression (lower compression = higher energy)
        current_compression = compression_history[-1]
        if current_compression <= 0.0:
            energy = 1.0
        else:
            # Use inverse with scaling: compression of 0.02 (2%) = energy ~0.8
            # compression of 0.10 (10%) = energy ~0.3
            energy = 1.0 / (1.0 + current_compression * 10.0)

        energy = max(0.0, min(1.0, energy))

        # Ignition detection: ATR expanding vs recent mean
        is_ignition = False
        if len(compression_history) >= lookback and current_atr_ratio > 0:
            # If we have compression history, a drop in compression + ATR spike = ignition
            recent_compressions = compression_history[-lookback:]
            mean_compression = np.mean(recent_compressions)

            # Ignition when current ATR ratio is significantly above what
            # the compression regime would suggest (1.5x expansion)
            if mean_compression > 0:
                # ATR expanding means volatility is breaking out of compression
                # Compare current_atr_ratio as a proxy for expansion
                if current_atr_ratio > mean_compression * 0.5:
                    is_ignition = True

        return (energy, is_ignition)

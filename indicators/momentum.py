"""
Momentum and trend structure analysis.

Market Theory:
Trend strength measured via EMA crossovers reveals the directional bias of the
market. When fast EMA is above slow EMA, buyers are in control and vice versa.
The normalized distance between EMAs quantifies trend conviction.

Momentum acceleration (slope of price via linear regression) captures whether
price is gaining or losing speed - a key differentiator between sustainable
trends and exhaustion.

Multi-timeframe alignment measures agreement across different time horizons.
When all timeframes agree on direction (MTF alignment = 1.0), the trend is
strong and continuation trades have higher probability. Divergence between
timeframes signals potential reversal or consolidation.
"""

from __future__ import annotations

import numpy as np


class MomentumAnalyzer:
    """Analyzes momentum structure and multi-timeframe alignment."""

    def _compute_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """Compute Exponential Moving Average.

        Standard EMA calculation using the multiplier 2/(period+1).

        Args:
            data: Array of price values.
            period: EMA period.

        Returns:
            Array of EMA values, same length as input.
            First (period-1) values use expanding window.
        """
        arr = np.asarray(data, dtype=np.float64)
        if len(arr) == 0:
            return arr

        multiplier = 2.0 / (period + 1.0)
        ema = np.empty_like(arr)
        ema[0] = arr[0]

        for i in range(1, len(arr)):
            ema[i] = arr[i] * multiplier + ema[i - 1] * (1.0 - multiplier)

        return ema

    def compute_trend_strength(
        self, closes: np.ndarray, fast_period: int = 8, slow_period: int = 21
    ) -> float:
        """Compute trend strength via normalized EMA difference.

        Formula: (EMA_fast - EMA_slow) / Close

        Positive values indicate bullish trend, negative indicates bearish.
        Magnitude represents strength of the trend.

        Args:
            closes: Array of closing prices (most recent last).
            fast_period: Period for fast EMA.
            slow_period: Period for slow EMA.

        Returns:
            Normalized EMA difference. Returns 0.0 if insufficient data.
        """
        if closes is None or len(closes) < slow_period:
            return 0.0

        arr = np.asarray(closes, dtype=np.float64)
        current_close = arr[-1]
        if current_close == 0.0:
            return 0.0

        ema_fast = self._compute_ema(arr, fast_period)
        ema_slow = self._compute_ema(arr, slow_period)

        trend = (ema_fast[-1] - ema_slow[-1]) / current_close
        return float(trend)

    def compute_momentum_acceleration(
        self, closes: np.ndarray, period: int = 14
    ) -> float:
        """Compute momentum acceleration via linear regression slope.

        Fits a linear regression to the most recent N closes and returns
        the slope normalized by the current price level.

        Args:
            closes: Array of closing prices (most recent last).
            period: Number of recent bars to fit regression on.

        Returns:
            Normalized slope. Positive = accelerating up, negative = decelerating.
            Returns 0.0 if insufficient data.
        """
        if closes is None or len(closes) < period:
            return 0.0

        arr = np.asarray(closes, dtype=np.float64)
        window = arr[-period:]
        current_close = window[-1]

        if current_close == 0.0:
            return 0.0

        # Linear regression using least squares
        x = np.arange(period, dtype=np.float64)
        x_mean = np.mean(x)
        y_mean = np.mean(window)

        numerator = np.sum((x - x_mean) * (window - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0.0:
            return 0.0

        slope = numerator / denominator

        # Normalize by price
        normalized_slope = slope / current_close
        return float(normalized_slope)

    def compute_mtf_alignment(
        self, trend_1m: float, trend_5m: float, trend_15m: float
    ) -> float:
        """Compute multi-timeframe alignment score.

        Measures agreement of trend directions across timeframes.
        All same direction = 1.0 (strong alignment).
        Mixed signals = 0.5 (conflicting).
        All opposing = 0.0 (maximum divergence - unlikely in practice).

        Args:
            trend_1m: Trend strength from 1-minute timeframe.
            trend_5m: Trend strength from 5-minute timeframe.
            trend_15m: Trend strength from 15-minute timeframe.

        Returns:
            Alignment score between 0.0 and 1.0.
        """
        trends = [trend_1m, trend_5m, trend_15m]

        # Count positive, negative, and neutral
        positive = sum(1 for t in trends if t > 0)
        negative = sum(1 for t in trends if t < 0)

        # Perfect alignment: all same direction
        if positive == 3 or negative == 3:
            return 1.0

        # Two agree, one neutral or opposed
        if positive == 2 or negative == 2:
            return 0.67

        # Mixed or all neutral
        if positive == 1 and negative == 1:
            return 0.33

        # All zero/neutral
        return 0.5

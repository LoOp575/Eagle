"""
Funding rate sentiment analysis.

Market Theory:
Perpetual futures funding rates reflect the cost of holding positions.
Positive funding means longs pay shorts (bullish crowd), negative means
shorts pay longs (bearish crowd).

When funding deviates significantly from its moving average (funding pressure),
it indicates one side is becoming crowded. Extreme crowding combined with
rising OI creates squeeze potential - the crowded side may be forced to
unwind, causing violent price moves in the opposite direction.

The Crowded Trade Index combines funding extremity with OI growth to quantify
how vulnerable the market is to a squeeze event.
"""

from __future__ import annotations

from typing import List

import numpy as np


class FundingAnalyzer:
    """Analyzes funding rates for sentiment and squeeze detection."""

    def compute_funding_pressure(
        self, funding_rates: List[float], period: int = 20
    ) -> float:
        """Compute funding pressure as deviation of current rate from its SMA.

        Positive pressure means funding is above average (longs crowded).
        Negative pressure means funding is below average (shorts crowded).

        Args:
            funding_rates: List of historical funding rates (most recent last).
            period: Lookback period for SMA calculation.

        Returns:
            Current funding rate minus SMA of rates. Returns 0.0 if
            insufficient data.
        """
        if funding_rates is None or len(funding_rates) == 0:
            return 0.0

        arr = np.asarray(funding_rates, dtype=np.float64)

        if len(arr) < period:
            # Use all available data if less than period
            sma = np.mean(arr)
        else:
            sma = np.mean(arr[-period:])

        current = arr[-1]
        return float(current - sma)

    def compute_crowded_trade_index(
        self, funding_rate: float, oi_change: float
    ) -> float:
        """Compute the Crowded Trade Index.

        Measures how crowded a trade is by combining the extremity of funding
        with the growth in open interest. High values indicate a large
        one-sided bet is building.

        Formula: CTI = abs(funding_rate) * oi_change

        Args:
            funding_rate: Current funding rate.
            oi_change: Percentage change in open interest (decimal).

        Returns:
            Crowded trade index value. Higher values indicate more crowding.
        """
        if funding_rate is None or oi_change is None:
            return 0.0
        return abs(funding_rate) * oi_change

    def detect_squeeze_potential(
        self, funding_pressure: float, oi_change: float
    ) -> float:
        """Detect the probability of a squeeze event (0 to 1).

        A squeeze is most likely when:
        - Funding is very negative (shorts are crowded and paying to hold)
        - OI is rising (more shorts are entering)
        OR
        - Funding is very positive (longs are crowded)
        - OI is rising (more longs are entering)

        The function uses a sigmoid-like mapping to produce a 0-1 probability.

        Args:
            funding_pressure: Current funding pressure (deviation from SMA).
            oi_change: Percentage change in open interest.

        Returns:
            Squeeze probability between 0.0 and 1.0.
        """
        if funding_pressure is None or oi_change is None:
            return 0.0

        # Squeeze potential is high when funding is extreme AND OI is growing
        # Use absolute funding pressure (direction doesn't matter for squeeze risk)
        funding_extremity = abs(funding_pressure)

        # Only consider rising OI as contributing to squeeze potential
        oi_factor = max(0.0, oi_change)

        # Raw squeeze score: product of extremity and OI growth
        # Scale factors chosen to produce meaningful 0-1 output
        # funding_extremity of 0.001 (0.1%) with oi_change of 0.05 (5%) = moderate
        raw_score = funding_extremity * oi_factor * 1000.0

        # Sigmoid-like clamping to 0-1 range
        # Using tanh for smooth saturation
        probability = float(np.tanh(raw_score))
        return max(0.0, min(1.0, probability))

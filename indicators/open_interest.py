"""
Open Interest microstructure analysis.

Market Theory:
Open Interest (OI) reflects the total number of outstanding derivative contracts.
Changes in OI combined with price action reveal market positioning:

- OI rising + price rising = new longs entering (BREAKOUT)
- OI rising + price sideways = accumulation phase (ACCUMULATION)
- OI rising + price falling = new shorts entering (SHORT_TRAP potential)
- OI falling + price rising = short covering (DISTRIBUTION)
- OI falling + price falling = long liquidations (CAPITULATION)

The second derivative of OI (acceleration) detects sudden changes in positioning
speed, which often precede major moves.
"""

from __future__ import annotations

from typing import List

import numpy as np


class OIAnalyzer:
    """Analyzes open interest dynamics for positioning intelligence."""

    # Price change threshold for "sideways" classification
    SIDEWAYS_THRESHOLD: float = 0.005

    def compute_oi_change(self, oi_current: float, oi_previous: float) -> float:
        """Compute percentage change in open interest.

        Args:
            oi_current: Current open interest value.
            oi_previous: Previous open interest value.

        Returns:
            Percentage change as a decimal (e.g., 0.05 = 5% increase).
            Returns 0.0 if previous OI is zero or invalid.
        """
        if oi_previous is None or oi_previous == 0.0:
            return 0.0
        if oi_current is None:
            return 0.0
        return (oi_current - oi_previous) / oi_previous

    def compute_oi_acceleration(self, oi_changes: List[float]) -> float:
        """Compute the acceleration (second derivative) of OI changes.

        This measures how quickly the rate of OI change is itself changing.
        A spike in acceleration often precedes a volatility event.

        Args:
            oi_changes: List of sequential OI percentage changes.

        Returns:
            Rate of change of the most recent OI change vs the previous.
            Returns 0.0 if insufficient data.
        """
        if oi_changes is None or len(oi_changes) < 2:
            return 0.0

        arr = np.asarray(oi_changes, dtype=np.float64)
        # Second derivative: difference of consecutive changes
        acceleration = arr[-1] - arr[-2]
        return float(acceleration)

    def classify_oi_regime(self, oi_change: float, price_change: float) -> str:
        """Classify the current OI regime based on OI and price movements.

        Uses the relationship between OI direction and price direction to
        identify the type of market activity occurring.

        Args:
            oi_change: Percentage change in open interest (decimal).
            price_change: Percentage change in price (decimal).

        Returns:
            One of: "ACCUMULATION", "BREAKOUT", "SHORT_TRAP",
            "DISTRIBUTION", "CAPITULATION", "NEUTRAL".
        """
        price_sideways = abs(price_change) < self.SIDEWAYS_THRESHOLD

        if oi_change > 0:
            if price_sideways:
                return "ACCUMULATION"
            elif price_change > 0:
                return "BREAKOUT"
            else:
                return "SHORT_TRAP"
        elif oi_change < 0:
            if price_sideways:
                return "NEUTRAL"
            elif price_change > 0:
                return "DISTRIBUTION"
            else:
                return "CAPITULATION"
        else:
            return "NEUTRAL"

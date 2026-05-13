"""
Market attention and relative ranking analysis.

Market Theory:
Attention is a finite resource in crypto markets. When a symbol captures
disproportionate volume or price action relative to the broader market,
it signals concentrated interest that often precedes explosive moves.

Relative Volume Rank shows where a symbol stands vs all monitored symbols
in terms of activity. Breakout Rank measures relative price momentum.
The combined Attention Score identifies symbols that are both active and
moving - the most likely candidates for imminent pumps or dumps.
"""

from __future__ import annotations

from typing import List

import numpy as np


class AttentionAnalyzer:
    """Analyzes relative market attention across symbols."""

    def compute_relative_volume_rank(
        self, symbol_volume: float, all_volumes: List[float]
    ) -> float:
        """Compute the percentile rank of a symbol's volume vs all symbols.

        Args:
            symbol_volume: The target symbol's volume metric.
            all_volumes: List of volume metrics for all monitored symbols.

        Returns:
            Percentile rank between 0.0 and 1.0.
            Returns 0.5 if no comparison data available.
        """
        if all_volumes is None or len(all_volumes) == 0:
            return 0.5
        if symbol_volume is None:
            return 0.0

        arr = np.asarray(all_volumes, dtype=np.float64)

        if len(arr) == 1:
            return 0.5

        # Percentile rank: fraction of values that are less than symbol_volume
        below = np.sum(arr < symbol_volume)
        rank = below / len(arr)
        return float(rank)

    def compute_breakout_rank(
        self, symbol_change: float, all_changes: List[float]
    ) -> float:
        """Compute the percentile rank of a symbol's price change vs all symbols.

        Args:
            symbol_change: The target symbol's price change metric.
            all_changes: List of price changes for all monitored symbols.

        Returns:
            Percentile rank between 0.0 and 1.0.
            Returns 0.5 if no comparison data available.
        """
        if all_changes is None or len(all_changes) == 0:
            return 0.5
        if symbol_change is None:
            return 0.0

        arr = np.asarray(all_changes, dtype=np.float64)

        if len(arr) == 1:
            return 0.5

        # Percentile rank: fraction of values that are less than symbol_change
        below = np.sum(arr < symbol_change)
        rank = below / len(arr)
        return float(rank)

    def compute_attention_score(
        self, volume_rank: float, breakout_rank: float
    ) -> float:
        """Compute combined attention score from volume and breakout ranks.

        Weighted average: 60% volume rank + 40% breakout rank.
        Volume is weighted higher because it is a leading indicator
        (volume precedes price).

        Args:
            volume_rank: Percentile rank of volume (0-1).
            breakout_rank: Percentile rank of price change (0-1).

        Returns:
            Combined attention score between 0.0 and 1.0.
        """
        vol_r = volume_rank if volume_rank is not None else 0.0
        brk_r = breakout_rank if breakout_rank is not None else 0.0

        score = 0.6 * vol_r + 0.4 * brk_r
        return max(0.0, min(1.0, score))

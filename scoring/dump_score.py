"""
Dump score calculation module.

Computes the probability of a downward price dump using a weighted
combination of bearish feature vector metrics with sigmoid normalization.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from config import ScoringConfig
from features.feature_vector import FeatureVector


class DumpScorer:
    """Calculates dump probability scores from feature vectors.

    Uses a weighted sum of bearish indicators with sigmoid normalization
    to produce a 0-100 score representing dump likelihood.
    """

    # Dump-specific weights
    OI_DROP_WEIGHT = 0.25
    FUNDING_CROWDING_WEIGHT = 0.25
    VOLUME_NO_GROWTH_WEIGHT = 0.20
    DISTRIBUTION_WEIGHT = 0.20
    LIQUIDITY_THIN_WEIGHT = 0.10

    def __init__(self, config: ScoringConfig) -> None:
        """Initialize with scoring configuration.

        Args:
            config: ScoringConfig for threshold values.
        """
        self._config = config

    def compute_dump_score(
        self, feature_vector: FeatureVector
    ) -> Tuple[float, List[str]]:
        """Compute dump probability score from a feature vector.

        Components:
            - OI drop pressure (0.25): Positions closing rapidly
            - Funding extreme crowding (0.25): Longs very crowded
            - Volume spike without price growth (0.20): Distribution signal
            - Distribution pattern score (0.20): OI regime analysis
            - Liquidity thinness (0.10): Thin liquidity amplifies dumps

        Args:
            feature_vector: Computed feature metrics for a symbol.

        Returns:
            Tuple of (normalized_score 0-100, list of contributing reasons).
        """
        reasons: List[str] = []

        # OI drop pressure: High when oi_change is very negative
        # Positions closing rapidly signals potential dump
        oi_drop = min(max(-feature_vector.oi_change / 0.15, 0.0), 1.0)
        if feature_vector.oi_change < -0.05:
            reasons.append(
                f"OI dropping rapidly ({feature_vector.oi_change * 100:.1f}% decline)"
            )

        # Funding extreme crowding: High when funding_pressure is extreme positive
        # Longs very crowded = vulnerable to liquidation cascade
        funding_crowding = min(
            max(feature_vector.funding_pressure / 0.5, 0.0), 1.0
        )
        if feature_vector.funding_pressure > 0.2:
            reasons.append(
                f"Extreme long crowding detected "
                f"(funding pressure={feature_vector.funding_pressure:.3f})"
            )

        # Volume spike without price growth: Distribution signal
        # High volume_z but low/negative momentum = selling into strength
        volume_without_growth = 0.0
        if feature_vector.volume_z > 1.5 and feature_vector.trend_strength < 0.2:
            volume_without_growth = min(feature_vector.volume_z / 3.0, 1.0)
            reasons.append(
                f"High volume without price growth "
                f"(vol_z={feature_vector.volume_z:.1f}, "
                f"trend={feature_vector.trend_strength:.2f})"
            )
        elif feature_vector.volume_z > 1.0 and feature_vector.trend_strength < 0.0:
            volume_without_growth = min(feature_vector.volume_z / 4.0, 1.0)
            reasons.append(
                f"Volume divergence from price "
                f"(vol_z={feature_vector.volume_z:.1f}, "
                f"trend={feature_vector.trend_strength:.2f})"
            )

        # Distribution pattern score: OI regime indicates distribution/capitulation
        distribution_score = 0.0
        if feature_vector.oi_regime == "DISTRIBUTION":
            distribution_score = 0.8
            reasons.append("OI regime indicates active distribution")
        elif feature_vector.oi_regime == "CAPITULATION":
            distribution_score = 1.0
            reasons.append("OI regime shows capitulation pattern")
        elif feature_vector.oi_regime == "SHORT_TRAP":
            distribution_score = 0.4
            reasons.append("Potential short trap forming")

        # Liquidity thinness: Direct from feature vector
        # Thin liquidity amplifies downward moves
        liquidity_thin = min(max(feature_vector.liquidity_thinness, 0.0), 1.0)
        if feature_vector.liquidity_thinness > 0.6:
            reasons.append(
                f"Thin liquidity amplifying risk "
                f"(thinness={feature_vector.liquidity_thinness:.2f})"
            )

        # Compute weighted raw score
        raw_score = (
            self.OI_DROP_WEIGHT * oi_drop
            + self.FUNDING_CROWDING_WEIGHT * funding_crowding
            + self.VOLUME_NO_GROWTH_WEIGHT * volume_without_growth
            + self.DISTRIBUTION_WEIGHT * distribution_score
            + self.LIQUIDITY_THIN_WEIGHT * liquidity_thin
        )

        # Sigmoid normalization to 0-100
        normalized_score = self._sigmoid_normalize(raw_score, k=6.0, midpoint=0.45)

        return normalized_score, reasons

    def _sigmoid_normalize(
        self, raw_score: float, k: float = 6.0, midpoint: float = 0.45
    ) -> float:
        """Apply sigmoid normalization to map raw score to 0-100.

        Uses the logistic function: score = 100 / (1 + exp(-k * (raw - midpoint)))

        Args:
            raw_score: Raw weighted sum (typically 0-1 range).
            k: Steepness of sigmoid curve.
            midpoint: Center point of the sigmoid where output = 50.

        Returns:
            Normalized score in 0-100 range.
        """
        try:
            exponent = -k * (raw_score - midpoint)
            exponent = max(min(exponent, 500.0), -500.0)
            return 100.0 / (1.0 + math.exp(exponent))
        except (OverflowError, ValueError):
            return 50.0

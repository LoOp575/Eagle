"""
Pump score calculation module.

Computes the probability of an upward price pump using a weighted
combination of feature vector metrics with sigmoid normalization.
"""

from __future__ import annotations

import math
from typing import List, Tuple

from config import ScoringConfig
from features.feature_vector import FeatureVector


class PumpScorer:
    """Calculates pump probability scores from feature vectors.

    Uses a weighted sum of bullish indicators with sigmoid normalization
    to produce a 0-100 score representing pump likelihood.
    """

    def __init__(self, config: ScoringConfig) -> None:
        """Initialize with scoring weights from configuration.

        Args:
            config: ScoringConfig containing weight parameters.
        """
        self._volume_z_weight = config.volume_z_weight
        self._oi_change_weight = config.oi_change_weight
        self._oi_acceleration_weight = config.oi_acceleration_weight
        self._compression_weight = config.compression_weight
        self._funding_weight = config.funding_weight
        self._momentum_weight = config.momentum_weight
        self._mcap_weight = config.mcap_weight

    def compute_pump_score(
        self, feature_vector: FeatureVector
    ) -> Tuple[float, List[str]]:
        """Compute pump probability score from a feature vector.

        Applies the weighted formula:
            raw = (Volume_Z * 0.25) + (OI_change * 0.20) +
                  (OI_acceleration * 0.10) + (Compression_inverse * 0.15) +
                  (Funding_pressure_inverse * 0.10) + (Momentum * 0.10) +
                  (MCAP_factor * 0.10)

        Then normalizes via sigmoid to 0-100 range.

        Args:
            feature_vector: Computed feature metrics for a symbol.

        Returns:
            Tuple of (normalized_score 0-100, list of contributing reasons).
        """
        reasons: List[str] = []

        # Volume Z-score component (higher = more bullish volume)
        volume_component = min(feature_vector.volume_z / 3.0, 1.0)
        volume_component = max(volume_component, 0.0)
        if feature_vector.volume_z > 2.0:
            reasons.append(
                f"Volume spike anomaly detected (z={feature_vector.volume_z:.1f})"
            )
        elif feature_vector.volume_z > 1.0:
            reasons.append(
                f"Above-average volume (z={feature_vector.volume_z:.1f})"
            )

        # OI change component (positive OI change = new positions opening)
        oi_component = min(max(feature_vector.oi_change / 0.15, 0.0), 1.0)
        if feature_vector.oi_change > 0.05:
            reasons.append(
                f"OI accumulation increasing (+{feature_vector.oi_change * 100:.1f}%)"
            )

        # OI acceleration (accelerating OI growth)
        oi_accel_component = min(
            max(feature_vector.oi_acceleration / 0.10, 0.0), 1.0
        )
        if feature_vector.oi_acceleration > 0.03:
            reasons.append(
                f"OI growth accelerating ({feature_vector.oi_acceleration * 100:.1f}%)"
            )

        # Compression inverse (low compression = tight range = potential breakout)
        compression_inverse = 1.0 - min(max(feature_vector.price_compression, 0.0), 1.0)
        if feature_vector.price_compression < 0.3:
            reasons.append(
                f"Price tightly compressed (ratio={feature_vector.price_compression:.2f})"
            )

        # Funding pressure inverse (negative funding = shorts paying longs = squeeze potential)
        funding_inverse = min(
            max(-feature_vector.funding_pressure / 0.5, 0.0), 1.0
        )
        if feature_vector.funding_pressure < -0.1:
            reasons.append(
                f"Negative funding indicating squeeze potential "
                f"(pressure={feature_vector.funding_pressure:.3f})"
            )

        # Momentum component (trend strength)
        momentum_component = min(max(feature_vector.trend_strength, 0.0), 1.0)
        if feature_vector.trend_strength > 0.6:
            reasons.append(
                f"Strong upward momentum (strength={feature_vector.trend_strength:.2f})"
            )

        # MCAP factor (higher = more susceptible to manipulation/pump)
        mcap_component = min(max(feature_vector.mcap_factor, 0.0), 1.0)
        if feature_vector.mcap_factor > 0.6:
            reasons.append(
                f"Low market cap susceptible to pump (factor={feature_vector.mcap_factor:.2f})"
            )

        # Compute weighted raw score
        raw_score = (
            self._volume_z_weight * volume_component
            + self._oi_change_weight * oi_component
            + self._oi_acceleration_weight * oi_accel_component
            + self._compression_weight * compression_inverse
            + self._funding_weight * funding_inverse
            + self._momentum_weight * momentum_component
            + self._mcap_weight * mcap_component
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
            k: Steepness of sigmoid curve. Higher = steeper transition.
            midpoint: Center point of the sigmoid where output = 50.

        Returns:
            Normalized score in 0-100 range.
        """
        try:
            exponent = -k * (raw_score - midpoint)
            # Clamp to prevent overflow
            exponent = max(min(exponent, 500.0), -500.0)
            return 100.0 / (1.0 + math.exp(exponent))
        except (OverflowError, ValueError):
            return 50.0

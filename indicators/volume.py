"""
Volume anomaly detection indicators.

Market Theory:
Volume precedes price. Abnormal volume spikes (measured via z-score) often signal
institutional positioning or smart-money activity before significant price moves.
A sudden volume surge in a low-float futures market is one of the strongest
leading indicators of an impending pump or dump.

The Volume Z-score measures how many standard deviations current volume is
from the recent mean. Values above 2.5 typically indicate anomalous activity.
Volume acceleration (change in z-score over time) helps detect the onset
of new volume regimes.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


class VolumeAnalyzer:
    """Analyzes volume patterns for anomaly detection and trend confirmation."""

    def compute_volume_z(self, volumes: np.ndarray, period: int = 20) -> float:
        """Compute the volume z-score: how many std devs current volume is from the mean.

        Formula: Z = (V_now - SMA(V, n)) / STD(V, n)

        Args:
            volumes: Array of volume values (most recent last).
            period: Lookback period for mean and std calculation.

        Returns:
            Z-score of the most recent volume bar. Returns 0.0 if insufficient data
            or zero standard deviation.
        """
        if volumes is None or len(volumes) < period:
            return 0.0

        arr = np.asarray(volumes, dtype=np.float64)
        window = arr[-period:]
        mean = np.mean(window)
        std = np.std(window, ddof=1)

        if std == 0.0 or np.isnan(std):
            return 0.0

        current = arr[-1]
        z_score = (current - mean) / std
        return float(z_score)

    def compute_volume_acceleration(
        self, volume_z_current: float, volume_z_previous: float
    ) -> float:
        """Compute volume acceleration as the difference between consecutive z-scores.

        Positive acceleration means volume is increasing at an accelerating rate,
        which is a stronger signal than a static high z-score.

        Args:
            volume_z_current: Current period's volume z-score.
            volume_z_previous: Previous period's volume z-score.

        Returns:
            Difference between current and previous z-scores.
        """
        return volume_z_current - volume_z_previous

    def detect_anomaly(
        self, volume_z: float, threshold: float = 2.5
    ) -> Tuple[bool, float]:
        """Detect whether the current volume constitutes an anomaly.

        An anomaly is defined as a volume z-score exceeding the specified threshold.
        The magnitude represents how far above the threshold the z-score is.

        Args:
            volume_z: Current volume z-score.
            threshold: Z-score threshold for anomaly detection (default 2.5).

        Returns:
            Tuple of (is_anomaly, magnitude). Magnitude is the absolute z-score
            value when anomaly is detected, 0.0 otherwise.
        """
        is_anomaly = abs(volume_z) >= threshold
        magnitude = abs(volume_z) if is_anomaly else 0.0
        return (is_anomaly, magnitude)

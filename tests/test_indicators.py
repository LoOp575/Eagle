"""
Unit tests for indicator modules.

Tests core indicator calculations including volume z-score,
open interest regime classification, funding pressure, compression,
momentum, and attention ranking.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from indicators.attention import AttentionAnalyzer
from indicators.compression import CompressionAnalyzer
from indicators.funding import FundingAnalyzer
from indicators.momentum import MomentumAnalyzer
from indicators.open_interest import OIAnalyzer
from indicators.volume import VolumeAnalyzer


class TestVolumeAnalyzer:
    """Tests for VolumeAnalyzer."""

    def test_volume_z_score_calculation(self) -> None:
        """Verify z-score math with known data."""
        analyzer = VolumeAnalyzer()

        # Create data where the last value is exactly 2 std devs above mean
        np.random.seed(10)
        base_volumes = np.ones(20) * 100.0
        # Make the last value a spike
        base_volumes[-1] = 200.0

        z = analyzer.compute_volume_z(base_volumes, period=20)

        # The mean of the window includes the spike
        # With 19 values at 100 and 1 at 200, mean = 105, std > 0
        # Z should be positive and significant
        assert z > 0.0, f"Expected positive z-score, got {z}"

    def test_volume_z_score_known_values(self) -> None:
        """Verify z-score with controlled data."""
        analyzer = VolumeAnalyzer()

        # 20 bars: first 19 at 100, last at 100 (no anomaly)
        flat_volumes = np.ones(20) * 100.0
        z_flat = analyzer.compute_volume_z(flat_volumes, period=20)
        # Std of uniform array (ddof=1) is 0, so should return 0
        assert z_flat == 0.0

    def test_volume_anomaly_detection(self) -> None:
        """Verify threshold-based anomaly detection."""
        analyzer = VolumeAnalyzer()

        # Z-score above threshold
        is_anomaly, magnitude = analyzer.detect_anomaly(3.0, threshold=2.5)
        assert is_anomaly is True
        assert magnitude == 3.0

        # Z-score below threshold
        is_anomaly, magnitude = analyzer.detect_anomaly(2.0, threshold=2.5)
        assert is_anomaly is False
        assert magnitude == 0.0

    def test_volume_acceleration(self) -> None:
        """Verify acceleration is difference of consecutive z-scores."""
        analyzer = VolumeAnalyzer()
        accel = analyzer.compute_volume_acceleration(3.0, 1.5)
        assert accel == 1.5


class TestOIAnalyzer:
    """Tests for OIAnalyzer."""

    def test_oi_change_computation(self) -> None:
        """Verify basic percentage change calculation."""
        analyzer = OIAnalyzer()

        # 10% increase
        change = analyzer.compute_oi_change(110.0, 100.0)
        assert abs(change - 0.10) < 1e-9

        # 5% decrease
        change = analyzer.compute_oi_change(95.0, 100.0)
        assert abs(change - (-0.05)) < 1e-9

        # Zero previous (edge case)
        change = analyzer.compute_oi_change(100.0, 0.0)
        assert change == 0.0

    def test_oi_regime_classification(self) -> None:
        """Test all 6 OI regime classifications."""
        analyzer = OIAnalyzer()

        # OI up + price up = BREAKOUT
        assert analyzer.classify_oi_regime(0.05, 0.02) == "BREAKOUT"

        # OI up + price sideways = ACCUMULATION
        assert analyzer.classify_oi_regime(0.05, 0.001) == "ACCUMULATION"

        # OI up + price down = SHORT_TRAP
        assert analyzer.classify_oi_regime(0.05, -0.02) == "SHORT_TRAP"

        # OI down + price up = DISTRIBUTION
        assert analyzer.classify_oi_regime(-0.05, 0.02) == "DISTRIBUTION"

        # OI down + price down = CAPITULATION
        assert analyzer.classify_oi_regime(-0.05, -0.02) == "CAPITULATION"

        # OI down + price sideways = NEUTRAL
        assert analyzer.classify_oi_regime(-0.05, 0.001) == "NEUTRAL"

        # OI unchanged = NEUTRAL
        assert analyzer.classify_oi_regime(0.0, 0.02) == "NEUTRAL"

    def test_oi_acceleration(self) -> None:
        """Verify OI acceleration calculation."""
        analyzer = OIAnalyzer()

        changes = [0.01, 0.02, 0.05]
        accel = analyzer.compute_oi_acceleration(changes)
        assert abs(accel - 0.03) < 1e-9


class TestFundingAnalyzer:
    """Tests for FundingAnalyzer."""

    def test_funding_pressure(self) -> None:
        """Verify SMA deviation calculation."""
        analyzer = FundingAnalyzer()

        # All rates the same: pressure should be 0
        rates = [0.0001] * 20
        pressure = analyzer.compute_funding_pressure(rates, period=20)
        assert abs(pressure) < 1e-10

        # Last rate much higher than average
        rates = [0.0001] * 19 + [0.0005]
        pressure = analyzer.compute_funding_pressure(rates, period=20)
        # current (0.0005) - mean of all 20 values
        expected_mean = (0.0001 * 19 + 0.0005) / 20
        expected_pressure = 0.0005 - expected_mean
        assert abs(pressure - expected_pressure) < 1e-10

    def test_squeeze_potential(self) -> None:
        """Verify squeeze potential returns 0-1 range."""
        analyzer = FundingAnalyzer()

        # Zero inputs
        prob = analyzer.detect_squeeze_potential(0.0, 0.0)
        assert prob == 0.0

        # High funding extremity + rising OI = high squeeze
        prob = analyzer.detect_squeeze_potential(0.5, 0.10)
        assert 0.0 <= prob <= 1.0
        assert prob > 0.5  # Should be high

        # Negative OI change = no squeeze potential
        prob = analyzer.detect_squeeze_potential(0.5, -0.10)
        assert prob == 0.0


class TestCompressionAnalyzer:
    """Tests for CompressionAnalyzer."""

    def test_compression_calculation(self) -> None:
        """Verify (H-L)/C formula."""
        analyzer = CompressionAnalyzer()

        # Simple case: high=110, low=90, close=100
        highs = np.array([110.0])
        lows = np.array([90.0])
        close = 100.0

        compression = analyzer.compute_price_compression(
            highs, lows, close, period=20
        )
        # (110-90)/100 = 0.20
        assert abs(compression - 0.20) < 1e-9

    def test_breakout_energy_ignition(self) -> None:
        """Test ignition detection from compression history."""
        analyzer = CompressionAnalyzer()

        # Low compression history (tightly compressed)
        compression_history = [0.02, 0.02, 0.02, 0.02, 0.02]
        # ATR expanding beyond compression threshold
        current_atr = 0.015

        energy, is_ignition = analyzer.detect_breakout_energy(
            compression_history, current_atr, lookback=5
        )

        # Energy should be high (inverse of low compression)
        assert energy > 0.5

        # Ignition: ATR ratio (0.015) > mean_compression (0.02) * 0.5 = 0.01
        assert is_ignition is True

    def test_atr_ratio(self) -> None:
        """Verify ATR ratio computation."""
        analyzer = CompressionAnalyzer()

        # Create simple price data
        highs = np.array([102.0, 103.0, 104.0, 103.0, 105.0])
        lows = np.array([98.0, 97.0, 99.0, 98.0, 100.0])
        closes = np.array([100.0, 101.0, 102.0, 100.0, 103.0])

        atr_ratio = analyzer.compute_atr_ratio(highs, lows, closes, period=14)
        assert atr_ratio > 0.0
        assert atr_ratio < 1.0  # Should be a small fraction of price


class TestMomentumAnalyzer:
    """Tests for MomentumAnalyzer."""

    def test_momentum_ema_trend(self) -> None:
        """Verify EMA crossover detection for trend strength."""
        analyzer = MomentumAnalyzer()

        # Uptrending prices: fast EMA should be above slow EMA
        closes = np.linspace(100.0, 120.0, 30)
        trend = analyzer.compute_trend_strength(closes, fast_period=8, slow_period=21)
        assert trend > 0.0, f"Expected positive trend, got {trend}"

        # Downtrending prices: fast EMA should be below slow EMA
        closes_down = np.linspace(120.0, 100.0, 30)
        trend_down = analyzer.compute_trend_strength(
            closes_down, fast_period=8, slow_period=21
        )
        assert trend_down < 0.0, f"Expected negative trend, got {trend_down}"

    def test_mtf_alignment_all_bullish(self) -> None:
        """All timeframes bullish: alignment score should be 1.0."""
        analyzer = MomentumAnalyzer()
        score = analyzer.compute_mtf_alignment(0.5, 0.3, 0.2)
        assert score == 1.0

    def test_mtf_alignment_mixed(self) -> None:
        """Mixed signals: alignment score should be intermediate."""
        analyzer = MomentumAnalyzer()

        # Two positive, one negative
        score = analyzer.compute_mtf_alignment(0.5, 0.3, -0.2)
        assert score == 0.67

        # One positive, one negative
        score = analyzer.compute_mtf_alignment(0.5, -0.3, 0.0)
        assert score == 0.33

    def test_mtf_alignment_all_bearish(self) -> None:
        """All timeframes bearish: alignment score should be 1.0."""
        analyzer = MomentumAnalyzer()
        score = analyzer.compute_mtf_alignment(-0.5, -0.3, -0.2)
        assert score == 1.0


class TestAttentionAnalyzer:
    """Tests for AttentionAnalyzer."""

    def test_attention_ranking(self) -> None:
        """Verify percentile rank calculation."""
        analyzer = AttentionAnalyzer()

        # Symbol volume is highest
        all_volumes = [100.0, 200.0, 300.0, 400.0, 500.0]
        rank = analyzer.compute_relative_volume_rank(500.0, all_volumes)
        # 4 out of 5 values are less than 500
        assert abs(rank - 0.8) < 1e-9

        # Symbol volume is lowest
        rank = analyzer.compute_relative_volume_rank(100.0, all_volumes)
        # 0 out of 5 values are less than 100
        assert abs(rank - 0.0) < 1e-9

    def test_attention_score_combined(self) -> None:
        """Verify combined attention score calculation."""
        analyzer = AttentionAnalyzer()

        # Perfect ranks
        score = analyzer.compute_attention_score(1.0, 1.0)
        assert abs(score - 1.0) < 1e-9

        # Zero ranks
        score = analyzer.compute_attention_score(0.0, 0.0)
        assert abs(score - 0.0) < 1e-9

        # Weighted: 0.6 * 0.8 + 0.4 * 0.5 = 0.48 + 0.20 = 0.68
        score = analyzer.compute_attention_score(0.8, 0.5)
        assert abs(score - 0.68) < 1e-9

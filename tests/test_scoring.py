"""
Unit tests for scoring modules.

Tests pump scorer, dump scorer, and decision engine including
weight application, sigmoid normalization bounds, and signal
generation thresholds.
"""

from __future__ import annotations

import sys

sys.path.insert(0, ".")

from config import ScoringConfig, TradingConfig
from features.feature_vector import FeatureVector
from scoring.decision_engine import DecisionEngine
from scoring.dump_score import DumpScorer
from scoring.pump_score import PumpScorer
from scoring.signal_models import Confidence, SignalType


def _make_scoring_config() -> ScoringConfig:
    """Create a scoring config for tests."""
    return ScoringConfig(
        volume_z_weight=0.25,
        oi_change_weight=0.20,
        oi_acceleration_weight=0.10,
        compression_weight=0.15,
        funding_weight=0.10,
        momentum_weight=0.10,
        mcap_weight=0.10,
    )


class TestPumpScorer:
    """Tests for PumpScorer."""

    def test_pump_score_high_signal(self) -> None:
        """Feature vector with strong bullish values should produce score > 75."""
        config = _make_scoring_config()
        scorer = PumpScorer(config)

        fv = FeatureVector(
            symbol="TEST/USDT",
            volume_z=3.0,
            oi_change=0.12,
            oi_acceleration=0.08,
            price_compression=0.1,
            funding_pressure=-0.3,
            trend_strength=0.9,
            mcap_factor=0.8,
        )

        score, reasons = scorer.compute_pump_score(fv)
        assert score > 75.0, f"Expected score > 75, got {score}"
        assert len(reasons) > 0

    def test_pump_score_low_signal(self) -> None:
        """Weak feature values should produce score < 30."""
        config = _make_scoring_config()
        scorer = PumpScorer(config)

        fv = FeatureVector(
            symbol="TEST/USDT",
            volume_z=0.2,
            oi_change=-0.01,
            oi_acceleration=-0.01,
            price_compression=0.8,
            funding_pressure=0.3,
            trend_strength=0.1,
            mcap_factor=0.1,
        )

        score, reasons = scorer.compute_pump_score(fv)
        assert score < 30.0, f"Expected score < 30, got {score}"

    def test_pump_score_weights_sum_to_one(self) -> None:
        """Verify that scoring weights sum to 1.0."""
        config = _make_scoring_config()
        total = (
            config.volume_z_weight
            + config.oi_change_weight
            + config.oi_acceleration_weight
            + config.compression_weight
            + config.funding_weight
            + config.momentum_weight
            + config.mcap_weight
        )
        assert abs(total - 1.0) < 1e-9, f"Weights sum to {total}, expected 1.0"

    def test_sigmoid_normalization_bounds(self) -> None:
        """Output should always be in 0-100 range."""
        config = _make_scoring_config()
        scorer = PumpScorer(config)

        # Test extreme high values
        fv_high = FeatureVector(
            symbol="TEST/USDT",
            volume_z=10.0,
            oi_change=1.0,
            oi_acceleration=1.0,
            price_compression=0.0,
            funding_pressure=-5.0,
            trend_strength=1.0,
            mcap_factor=1.0,
        )
        score_high, _ = scorer.compute_pump_score(fv_high)
        assert 0.0 <= score_high <= 100.0

        # Test extreme low values
        fv_low = FeatureVector(
            symbol="TEST/USDT",
            volume_z=-5.0,
            oi_change=-1.0,
            oi_acceleration=-1.0,
            price_compression=5.0,
            funding_pressure=5.0,
            trend_strength=-1.0,
            mcap_factor=-1.0,
        )
        score_low, _ = scorer.compute_pump_score(fv_low)
        assert 0.0 <= score_low <= 100.0


class TestDumpScorer:
    """Tests for DumpScorer."""

    def test_dump_score_high_risk(self) -> None:
        """Distribution pattern should produce high dump score."""
        config = _make_scoring_config()
        scorer = DumpScorer(config)

        fv = FeatureVector(
            symbol="TEST/USDT",
            oi_change=-0.10,
            funding_pressure=0.4,
            volume_z=2.5,
            trend_strength=0.05,
            oi_regime="DISTRIBUTION",
            liquidity_thinness=0.8,
        )

        score, reasons = scorer.compute_dump_score(fv)
        assert score > 60.0, f"Expected score > 60, got {score}"
        assert len(reasons) > 0

    def test_dump_score_low_risk(self) -> None:
        """Healthy market should produce low dump score."""
        config = _make_scoring_config()
        scorer = DumpScorer(config)

        fv = FeatureVector(
            symbol="TEST/USDT",
            oi_change=0.05,
            funding_pressure=0.0,
            volume_z=0.5,
            trend_strength=0.6,
            oi_regime="ACCUMULATION",
            liquidity_thinness=0.2,
        )

        score, reasons = scorer.compute_dump_score(fv)
        assert score < 50.0, f"Expected score < 50, got {score}"


class TestDecisionEngine:
    """Tests for DecisionEngine."""

    def test_decision_engine_long_signal(self) -> None:
        """pump > 75 and dump < 40 should produce LONG signal."""
        config = _make_scoring_config()
        engine = DecisionEngine(config)

        fv = FeatureVector(symbol="TEST/USDT", atr_ratio=0.02)
        signal = engine.generate_signal(
            pump_score=80.0,
            dump_score=25.0,
            pump_reasons=["Volume spike"],
            dump_reasons=[],
            feature_vector=fv,
        )

        assert signal.signal_type == SignalType.HIGH_PROBABILITY_LONG
        assert signal.confidence == Confidence.HIGH

    def test_decision_engine_short_signal(self) -> None:
        """dump > 70 and pump < 40 should produce SHORT signal."""
        config = _make_scoring_config()
        engine = DecisionEngine(config)

        fv = FeatureVector(symbol="TEST/USDT", atr_ratio=0.02)
        signal = engine.generate_signal(
            pump_score=30.0,
            dump_score=75.0,
            pump_reasons=[],
            dump_reasons=["OI dropping"],
            feature_vector=fv,
        )

        assert signal.signal_type == SignalType.HIGH_RISK_DUMP_SHORT
        assert signal.confidence == Confidence.HIGH

    def test_decision_engine_watch(self) -> None:
        """pump 50-75 with dump > 50 should produce WATCH signal."""
        config = _make_scoring_config()
        engine = DecisionEngine(config)

        fv = FeatureVector(symbol="TEST/USDT", atr_ratio=0.02)
        signal = engine.generate_signal(
            pump_score=55.0,
            dump_score=55.0,
            pump_reasons=["Moderate volume"],
            dump_reasons=[],
            feature_vector=fv,
        )

        assert signal.signal_type == SignalType.EARLY_ACCUMULATION_WATCH

    def test_decision_engine_no_trade(self) -> None:
        """Low scores should produce NO_TRADE signal."""
        config = _make_scoring_config()
        engine = DecisionEngine(config)

        fv = FeatureVector(symbol="TEST/USDT", atr_ratio=0.02)
        signal = engine.generate_signal(
            pump_score=30.0,
            dump_score=30.0,
            pump_reasons=[],
            dump_reasons=[],
            feature_vector=fv,
        )

        assert signal.signal_type == SignalType.NO_TRADE
        assert signal.confidence == Confidence.LOW

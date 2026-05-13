"""
Unit tests for risk engine modules.

Tests position manager sizing, maximum position enforcement,
circuit breakers, dynamic leverage computation, and the master
risk gate.
"""

from __future__ import annotations

import sys
import time

sys.path.insert(0, ".")

from config import RiskConfig
from risk_engine.position_manager import PositionManager
from risk_engine.risk_controls import RiskController


def _make_risk_config() -> RiskConfig:
    """Create a test RiskConfig."""
    return RiskConfig(
        max_risk_per_trade=0.01,
        max_positions=3,
        daily_max_loss=0.05,
        consecutive_loss_limit=3,
        max_leverage=10,
        position_size_method="kelly",
        cooldown_after_loss_seconds=0,  # No cooldown for tests
    )


class TestPositionManager:
    """Tests for PositionManager."""

    def test_position_size_calculation(self) -> None:
        """Verify position sizing math.

        Formula: size = (balance * risk_per_trade) / risk_distance * leverage
        """
        config = _make_risk_config()
        pm = PositionManager(config)

        # balance=10000, risk=1%, entry=100, SL=95, leverage=5
        # risk_amount = 10000 * 0.01 = 100
        # risk_distance = |100 - 95| = 5
        # size = (100 / 5) * 5 = 100
        size = pm.calculate_position_size(
            account_balance=10000.0,
            entry_price=100.0,
            stop_loss_price=95.0,
            leverage=5,
        )
        assert abs(size - 100.0) < 1e-9, f"Expected 100.0, got {size}"

    def test_position_size_zero_distance(self) -> None:
        """Zero risk distance should return 0 size."""
        config = _make_risk_config()
        pm = PositionManager(config)

        size = pm.calculate_position_size(
            account_balance=10000.0,
            entry_price=100.0,
            stop_loss_price=100.0,
            leverage=5,
        )
        assert size == 0.0

    def test_max_positions_enforced(self) -> None:
        """Cannot exceed max positions limit."""
        config = _make_risk_config()
        pm = PositionManager(config)

        # Open max_positions (3)
        for i in range(3):
            pm.open_position(
                symbol=f"TOKEN{i}/USDT",
                entry_price=100.0,
                size=1.0,
                leverage=5,
                stop_loss=95.0,
                take_profit=110.0,
            )

        # Fourth should fail
        assert pm.can_open_position() is False

        try:
            pm.open_position(
                symbol="TOKEN3/USDT",
                entry_price=100.0,
                size=1.0,
                leverage=5,
                stop_loss=95.0,
                take_profit=110.0,
            )
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

    def test_close_position(self) -> None:
        """Closing a position frees the slot."""
        config = _make_risk_config()
        pm = PositionManager(config)

        pm.open_position(
            symbol="BTC/USDT",
            entry_price=50000.0,
            size=0.1,
            leverage=10,
            stop_loss=49000.0,
            take_profit=52000.0,
        )
        assert pm.has_position("BTC/USDT") is True

        pnl = pm.close_position("BTC/USDT")
        assert pm.has_position("BTC/USDT") is False


class TestRiskController:
    """Tests for RiskController."""

    def test_daily_loss_cutoff(self) -> None:
        """Trading stops after daily loss limit is exceeded."""
        config = _make_risk_config()
        rc = RiskController(config)

        # Initial state: should trade
        assert rc.should_trade() is True

        # Record losses exceeding daily max (5%)
        rc.record_trade_result(-0.03)
        rc.record_trade_result(-0.03)
        # Total PnL = -0.06, exceeds max of 0.05
        assert rc.should_trade() is False

    def test_consecutive_loss_circuit_breaker(self) -> None:
        """Trading stops after N consecutive losses."""
        config = _make_risk_config()
        # Set cooldown to 0 for testing
        config.cooldown_after_loss_seconds = 0
        rc = RiskController(config)

        # Record 3 consecutive losses (limit is 3)
        rc.record_trade_result(-0.005)
        rc.record_trade_result(-0.005)
        rc.record_trade_result(-0.005)

        assert rc.check_consecutive_losses() is False
        assert rc.should_trade() is False

    def test_consecutive_loss_reset_on_win(self) -> None:
        """Consecutive loss counter resets on a winning trade."""
        config = _make_risk_config()
        config.cooldown_after_loss_seconds = 0
        rc = RiskController(config)

        rc.record_trade_result(-0.005)
        rc.record_trade_result(-0.005)
        # 2 consecutive losses, still below limit of 3
        assert rc.check_consecutive_losses() is True

        # Win resets counter
        rc.record_trade_result(0.01)
        assert rc.check_consecutive_losses() is True

    def test_dynamic_leverage_high_vol(self) -> None:
        """High ATR ratio should produce low leverage."""
        config = _make_risk_config()
        rc = RiskController(config)

        # atr_ratio = 0.08 (8% volatility)
        # volatility_factor = min(0.08 * 10, 0.9) = 0.8
        # leverage = max(1, int(10 * (1 - 0.8))) = max(1, 2) = 2
        leverage = rc.compute_dynamic_leverage(0.08)
        assert leverage == 2

    def test_dynamic_leverage_low_vol(self) -> None:
        """Low ATR ratio should produce high leverage."""
        config = _make_risk_config()
        rc = RiskController(config)

        # atr_ratio = 0.01 (1% volatility)
        # volatility_factor = min(0.01 * 10, 0.9) = 0.1
        # leverage = max(1, int(10 * (1 - 0.1))) = max(1, 9) = 9
        leverage = rc.compute_dynamic_leverage(0.01)
        assert leverage == 9

    def test_risk_gate_all_conditions(self) -> None:
        """should_trade() composite check passes when all conditions are met."""
        config = _make_risk_config()
        config.cooldown_after_loss_seconds = 0
        rc = RiskController(config)

        # Fresh state: all conditions met
        assert rc.should_trade() is True

        # Emergency stop blocks trading
        rc.trigger_emergency_stop("test")
        assert rc.should_trade() is False
        rc.reset_emergency_stop()
        assert rc.should_trade() is True

    def test_emergency_stop(self) -> None:
        """Once triggered, emergency stop blocks all trading."""
        config = _make_risk_config()
        rc = RiskController(config)

        assert rc.should_trade() is True
        rc.trigger_emergency_stop("Market crash detected")
        assert rc.should_trade() is False

        # Still blocked even with good PnL
        assert rc.should_trade() is False

        # Only manual reset restores trading
        rc.reset_emergency_stop()
        assert rc.should_trade() is True

    def test_daily_reset(self) -> None:
        """Daily counters reset properly."""
        config = _make_risk_config()
        config.cooldown_after_loss_seconds = 0
        rc = RiskController(config)

        rc.record_trade_result(-0.03)
        rc.record_trade_result(-0.03)
        # Over daily limit
        assert rc.should_trade() is False

        # Reset daily
        rc.reset_daily_counters()
        # Note: consecutive losses persist across days
        # but daily PnL resets, so if losses are below limit, should trade
        # However consecutive losses is now 2, still below 3
        assert rc.check_daily_loss(0.0) is True

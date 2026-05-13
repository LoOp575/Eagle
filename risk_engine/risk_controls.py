"""
Risk controls and circuit breakers module.

Implements daily loss limits, consecutive loss tracking, dynamic leverage
calculation, and emergency stop functionality to protect trading capital.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from config import RiskConfig


class RiskController:
    """Enforces risk management rules and circuit breakers.

    Tracks daily PnL, consecutive losses, and provides a master gate
    that must be checked before any trade execution. Implements dynamic
    leverage scaling based on market volatility.
    """

    def __init__(self, config: RiskConfig) -> None:
        """Initialize risk controller with configuration.

        Args:
            config: RiskConfig with risk limits and parameters.
        """
        self._daily_max_loss = config.daily_max_loss
        self._consecutive_loss_limit = config.consecutive_loss_limit
        self._max_leverage = config.max_leverage
        self._cooldown_seconds = config.cooldown_after_loss_seconds

        # Tracking state
        self._daily_pnl: float = 0.0
        self._consecutive_losses: int = 0
        self._trade_count_today: int = 0
        self._emergency_stop: bool = False
        self._emergency_reason: str = ""
        self._last_reset_date: Optional[str] = None
        self._last_trade_time: float = 0.0

    def check_daily_loss(self, current_pnl: float) -> bool:
        """Check if daily loss is within acceptable limits.

        Args:
            current_pnl: Current cumulative daily PnL as a fraction
                        (e.g., -0.03 means 3% loss).

        Returns:
            True if within limits (trading allowed), False if limit breached.
        """
        return current_pnl > -self._daily_max_loss

    def check_consecutive_losses(self) -> bool:
        """Check if consecutive loss count is below limit.

        Returns:
            True if below limit (trading allowed), False if limit reached.
        """
        return self._consecutive_losses < self._consecutive_loss_limit

    def record_trade_result(self, pnl: float) -> None:
        """Record the result of a completed trade.

        Updates daily PnL, consecutive loss counter, and trade count.
        Resets consecutive losses on a winning trade.

        Args:
            pnl: Profit/loss from the trade (positive = win, negative = loss).
        """
        self._daily_pnl += pnl
        self._trade_count_today += 1
        self._last_trade_time = time.time()

        if pnl < 0.0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

    def compute_dynamic_leverage(
        self, atr_ratio: float, max_leverage: int = 0
    ) -> int:
        """Compute appropriate leverage based on market volatility.

        Higher volatility results in lower leverage to manage risk.
        Formula: max(1, int(max_leverage * (1 - min(atr_ratio * 10, 0.9))))

        Args:
            atr_ratio: ATR as a ratio of price (e.g., 0.05 = 5% ATR).
            max_leverage: Maximum allowed leverage. Uses config default if 0.

        Returns:
            Recommended leverage as integer (1 to max_leverage).
        """
        if max_leverage <= 0:
            max_leverage = self._max_leverage

        # Scale down leverage as volatility increases
        volatility_factor = min(atr_ratio * 10.0, 0.9)
        leverage = max(1, int(max_leverage * (1.0 - volatility_factor)))

        return leverage

    def should_trade(self) -> bool:
        """Master gate check for trade permission.

        Checks all risk conditions:
            - Emergency stop not active
            - Daily loss within limits
            - Consecutive losses below limit
            - Cooldown period after last loss respected

        Returns:
            True if all conditions met and trading is allowed.
        """
        # Emergency stop overrides everything
        if self._emergency_stop:
            return False

        # Check daily loss limit
        if not self.check_daily_loss(self._daily_pnl):
            return False

        # Check consecutive losses
        if not self.check_consecutive_losses():
            return False

        # Check cooldown after loss
        if self._consecutive_losses > 0:
            elapsed = time.time() - self._last_trade_time
            if elapsed < self._cooldown_seconds:
                return False

        return True

    def trigger_emergency_stop(self, reason: str) -> None:
        """Activate emergency stop - prevents all trading.

        Args:
            reason: Description of why emergency stop was triggered.
        """
        self._emergency_stop = True
        self._emergency_reason = reason

    def reset_emergency_stop(self) -> None:
        """Deactivate emergency stop to resume trading."""
        self._emergency_stop = False
        self._emergency_reason = ""

    def reset_daily_counters(self) -> None:
        """Reset daily tracking counters.

        Should be called at the start of each new trading day.
        Resets daily PnL and trade count but preserves consecutive
        loss counter across days.
        """
        self._daily_pnl = 0.0
        self._trade_count_today = 0
        self._last_reset_date = time.strftime("%Y-%m-%d")

    def get_risk_status(self) -> dict:
        """Get current state of all risk metrics.

        Returns:
            Dictionary with all risk tracking information.
        """
        return {
            "daily_pnl": self._daily_pnl,
            "daily_max_loss": self._daily_max_loss,
            "consecutive_losses": self._consecutive_losses,
            "consecutive_loss_limit": self._consecutive_loss_limit,
            "trade_count_today": self._trade_count_today,
            "emergency_stop": self._emergency_stop,
            "emergency_reason": self._emergency_reason,
            "last_reset_date": self._last_reset_date,
            "should_trade": self.should_trade(),
            "max_leverage": self._max_leverage,
        }

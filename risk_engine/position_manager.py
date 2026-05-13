"""
Position management module.

Tracks open positions, computes position sizes based on risk parameters,
and enforces maximum position limits.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config import RiskConfig


@dataclass
class PositionInfo:
    """Information about an open trading position."""

    symbol: str
    entry_price: float
    size: float
    leverage: int
    stop_loss: float
    take_profit: float
    entry_time: float = field(default_factory=time.time)
    unrealized_pnl: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "symbol": self.symbol,
            "entry_price": self.entry_price,
            "size": self.size,
            "leverage": self.leverage,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "entry_time": self.entry_time,
            "unrealized_pnl": self.unrealized_pnl,
        }


class PositionManager:
    """Manages open trading positions and position sizing.

    Tracks all open positions, enforces maximum position limits,
    computes position sizes based on risk parameters, and provides
    portfolio-level exposure metrics.
    """

    def __init__(self, config: RiskConfig) -> None:
        """Initialize position manager with risk configuration.

        Args:
            config: RiskConfig containing position limits and risk params.
        """
        self._max_positions = config.max_positions
        self._risk_per_trade = config.max_risk_per_trade
        self._open_positions: Dict[str, PositionInfo] = {}

    def calculate_position_size(
        self,
        account_balance: float,
        entry_price: float,
        stop_loss_price: float,
        leverage: int,
    ) -> float:
        """Calculate position size based on fixed fractional risk.

        Formula: size = (balance * risk_per_trade) / |entry - stop_loss| * leverage

        This ensures that if the stop loss is hit, the maximum loss is
        limited to risk_per_trade percent of account balance.

        Args:
            account_balance: Current account equity in USD.
            entry_price: Planned entry price.
            stop_loss_price: Stop loss price level.
            leverage: Leverage multiplier to apply.

        Returns:
            Position size in base asset units.
        """
        if entry_price <= 0.0 or account_balance <= 0.0:
            return 0.0

        risk_distance = abs(entry_price - stop_loss_price)
        if risk_distance <= 0.0:
            return 0.0

        # Risk amount in USD
        risk_amount = account_balance * self._risk_per_trade

        # Position size = risk_amount / risk_per_unit * leverage
        position_size = (risk_amount / risk_distance) * leverage

        return position_size

    def can_open_position(self) -> bool:
        """Check if a new position can be opened.

        Returns:
            True if current open positions is below the maximum limit.
        """
        return len(self._open_positions) < self._max_positions

    def open_position(
        self,
        symbol: str,
        entry_price: float,
        size: float,
        leverage: int,
        stop_loss: float,
        take_profit: float,
    ) -> PositionInfo:
        """Record a new open position.

        Args:
            symbol: Trading pair symbol.
            entry_price: Actual entry price.
            size: Position size in base asset.
            leverage: Leverage used.
            stop_loss: Stop loss price level.
            take_profit: Take profit price level.

        Returns:
            PositionInfo for the new position.

        Raises:
            ValueError: If max positions exceeded or symbol already open.
        """
        if not self.can_open_position():
            raise ValueError(
                f"Cannot open position: max positions ({self._max_positions}) reached"
            )

        if symbol in self._open_positions:
            raise ValueError(f"Position already open for {symbol}")

        position = PositionInfo(
            symbol=symbol,
            entry_price=entry_price,
            size=size,
            leverage=leverage,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=time.time(),
            unrealized_pnl=0.0,
        )

        self._open_positions[symbol] = position
        return position

    def close_position(self, symbol: str) -> Optional[float]:
        """Close an open position and return realized PnL.

        Args:
            symbol: Trading pair symbol to close.

        Returns:
            Realized PnL (unrealized at close time), or None if no position.
        """
        position = self._open_positions.pop(symbol, None)
        if position is None:
            return None
        return position.unrealized_pnl

    def update_position_pnl(self, symbol: str, current_price: float) -> None:
        """Update unrealized PnL for an open position.

        Args:
            symbol: Trading pair symbol.
            current_price: Current market price.
        """
        position = self._open_positions.get(symbol)
        if position is None:
            return

        # PnL = (current_price - entry_price) * size * leverage
        price_diff = current_price - position.entry_price
        position.unrealized_pnl = price_diff * position.size * position.leverage

    def get_open_positions(self) -> List[PositionInfo]:
        """Get list of all open positions.

        Returns:
            List of PositionInfo for all open positions.
        """
        return list(self._open_positions.values())

    def get_total_exposure(self) -> float:
        """Calculate total portfolio exposure.

        Returns:
            Sum of all position sizes multiplied by their leverage.
        """
        return sum(
            pos.size * pos.leverage for pos in self._open_positions.values()
        )

    def has_position(self, symbol: str) -> bool:
        """Check if a position is already open for a symbol.

        Args:
            symbol: Trading pair symbol to check.

        Returns:
            True if position exists for the symbol.
        """
        return symbol in self._open_positions

    def get_position(self, symbol: str) -> Optional[PositionInfo]:
        """Get position info for a specific symbol.

        Args:
            symbol: Trading pair symbol.

        Returns:
            PositionInfo if position exists, None otherwise.
        """
        return self._open_positions.get(symbol)

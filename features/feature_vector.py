"""
Feature vector data model.

Contains all computed metrics for a single symbol at a point in time.
This is the output of the FeatureEngine and the input to the ScoringEngine.
Each field represents one computed indicator that contributes to the
pump/dump probability calculation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FeatureVector:
    """Complete feature vector for a single symbol.

    Contains all computed indicator values needed by the scoring module
    to calculate pump/dump probabilities and generate trading signals.
    """

    # Identification
    symbol: str = ""
    timestamp: Optional[int] = None

    # Volume indicators
    volume_z: float = 0.0
    volume_acceleration: float = 0.0
    volume_anomaly: bool = False

    # Open interest indicators
    oi_change: float = 0.0
    oi_acceleration: float = 0.0
    oi_regime: str = "NEUTRAL"

    # Funding indicators
    funding_pressure: float = 0.0
    crowded_trade_index: float = 0.0
    squeeze_potential: float = 0.0

    # Price compression indicators
    price_compression: float = 0.0
    atr_ratio: float = 0.0
    breakout_energy: float = 0.0
    is_ignition: bool = False

    # Liquidity indicators
    mcap_factor: float = 0.0
    orderbook_depth: float = 0.0
    liquidity_thinness: float = 0.0

    # Momentum indicators
    trend_strength: float = 0.0
    momentum_acceleration: float = 0.0
    mtf_alignment: float = 0.5

    # Attention/ranking indicators
    volume_rank: float = 0.5
    breakout_rank: float = 0.5
    attention_score: float = 0.5

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "volume_z": self.volume_z,
            "volume_acceleration": self.volume_acceleration,
            "volume_anomaly": self.volume_anomaly,
            "oi_change": self.oi_change,
            "oi_acceleration": self.oi_acceleration,
            "oi_regime": self.oi_regime,
            "funding_pressure": self.funding_pressure,
            "crowded_trade_index": self.crowded_trade_index,
            "squeeze_potential": self.squeeze_potential,
            "price_compression": self.price_compression,
            "atr_ratio": self.atr_ratio,
            "breakout_energy": self.breakout_energy,
            "is_ignition": self.is_ignition,
            "mcap_factor": self.mcap_factor,
            "orderbook_depth": self.orderbook_depth,
            "liquidity_thinness": self.liquidity_thinness,
            "trend_strength": self.trend_strength,
            "momentum_acceleration": self.momentum_acceleration,
            "mtf_alignment": self.mtf_alignment,
            "volume_rank": self.volume_rank,
            "breakout_rank": self.breakout_rank,
            "attention_score": self.attention_score,
        }

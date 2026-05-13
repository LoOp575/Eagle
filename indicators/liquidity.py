"""
Liquidity and market capitalization analysis.

Market Theory:
Low-liquidity markets are more susceptible to manipulation and violent price
moves. The market cap factor estimates how easy it is to move price - smaller
cap assets require less capital to create significant price impact.

Order book depth analysis reveals real-time liquidity conditions. Thin order
books (low depth relative to volume) indicate vulnerability to sudden moves.
The liquidity thinness ratio captures the relationship between available
liquidity and recent trading activity - high thinness means even moderate
volume can cause outsized price impact.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np


class LiquidityAnalyzer:
    """Analyzes liquidity conditions and manipulation susceptibility."""

    def compute_mcap_factor(
        self,
        volume_24h: float,
        price: float,
        circulating_supply: float = None,
    ) -> float:
        """Compute market cap manipulation susceptibility factor.

        Uses 1/log(market_cap) to score how easy it is to move the asset.
        Smaller market cap = higher factor = easier to manipulate.

        If circulating supply is not available, estimates market cap from
        24h volume using a heuristic (volume typically 5-20% of mcap).

        Args:
            volume_24h: 24-hour trading volume in quote currency.
            price: Current price of the asset.
            circulating_supply: Known circulating supply (optional).

        Returns:
            Factor clamped to 0-1 range. Higher values indicate greater
            susceptibility to price manipulation.
        """
        if volume_24h is None or volume_24h <= 0 or price is None or price <= 0:
            return 0.0

        if circulating_supply is not None and circulating_supply > 0:
            market_cap = price * circulating_supply
        else:
            # Estimate market cap from volume
            # Heuristic: daily volume is roughly 10% of market cap for liquid futures
            market_cap = volume_24h * 10.0

        if market_cap <= 1.0:
            return 1.0

        # 1/log(mcap) gives higher values for smaller caps
        log_mcap = math.log(market_cap)
        if log_mcap <= 0:
            return 1.0

        # Normalize: log(1e6)=13.8, log(1e9)=20.7, log(1e12)=27.6
        # We want small caps (1e6-1e8) to score high (0.7-1.0)
        # and large caps (1e10+) to score low (0.0-0.3)
        raw_factor = 1.0 / log_mcap

        # Scale to make values more spread in useful range
        # Multiply by ~20 so that log_mcap=14 -> ~1.4 -> clamp 1.0
        # and log_mcap=25 -> ~0.8
        scaled = raw_factor * 20.0

        return max(0.0, min(1.0, scaled))

    def compute_orderbook_depth(
        self, bids: List, asks: List, levels: int = 10
    ) -> float:
        """Compute order book depth score.

        Sums bid and ask volume in the top N levels to gauge liquidity.
        Lower depth means easier to move price, resulting in a higher score
        for pump/dump potential.

        Args:
            bids: List of bid levels. Each can be [price, qty] or an object
                  with a .quantity attribute.
            asks: List of ask levels. Same format as bids.
            levels: Number of top levels to include.

        Returns:
            Depth score normalized to 0-1. Higher score = thinner book
            (more vulnerable to manipulation).
        """
        if (bids is None or len(bids) == 0) and (asks is None or len(asks) == 0):
            return 1.0  # No book data = assume thin

        total_depth = 0.0

        if bids is not None:
            for level in bids[:levels]:
                if hasattr(level, "quantity"):
                    total_depth += level.quantity
                elif isinstance(level, (list, tuple)) and len(level) >= 2:
                    total_depth += float(level[1])

        if asks is not None:
            for level in asks[:levels]:
                if hasattr(level, "quantity"):
                    total_depth += level.quantity
                elif isinstance(level, (list, tuple)) and len(level) >= 2:
                    total_depth += float(level[1])

        if total_depth <= 0:
            return 1.0

        # Normalize depth to 0-1 using log scale
        # Deeper books = lower score (less manipulable)
        # Use log to handle wide range of depths across assets
        log_depth = math.log1p(total_depth)

        # Typical depth ranges: 10-10000 units
        # log1p(10)=2.4, log1p(100)=4.6, log1p(1000)=6.9, log1p(10000)=9.2
        # Invert and normalize: higher raw depth = lower score
        max_log = 12.0  # log1p(~160000) - very deep book
        score = 1.0 - (log_depth / max_log)

        return max(0.0, min(1.0, score))

    def compute_liquidity_thinness(
        self, orderbook_depth: float, volume_24h: float
    ) -> float:
        """Compute liquidity thinness ratio.

        Measures how thin the order book is relative to recent trading volume.
        Higher values mean the available liquidity is thin compared to the
        volume flowing through the market, making it more prone to slippage
        and manipulation.

        Args:
            orderbook_depth: Depth score from compute_orderbook_depth (0-1,
                             higher = thinner).
            volume_24h: 24-hour trading volume.

        Returns:
            Liquidity thinness score. Higher = thinner = more pump/dump prone.
            Returns 0.0 if volume data is missing.
        """
        if volume_24h is None or volume_24h <= 0:
            return 0.0

        # orderbook_depth is already normalized (higher = thinner book)
        # We weight it by volume to get "effective thinness"
        # High volume + thin book = very thin (dangerous)
        # Low volume + thin book = less concerning

        # Normalize volume using log scale
        log_vol = math.log1p(volume_24h)
        # Typical 24h volumes: 1e5 to 1e9
        # log1p(1e5)=11.5, log1p(1e7)=16.1, log1p(1e9)=20.7
        vol_factor = log_vol / 25.0  # normalize to rough 0-1 range

        # Thinness = depth_score * vol_factor
        # Thin book + high volume = very vulnerable
        thinness = orderbook_depth * vol_factor

        return max(0.0, min(1.0, thinness))

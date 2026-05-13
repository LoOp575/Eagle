"""
Decision engine for generating trade signals.

Takes pump and dump scores along with feature vectors and produces
actionable trading signals with confidence levels and trade parameters.
"""

from __future__ import annotations

import time
from typing import List, Optional

from config import ScoringConfig, TradingConfig
from features.feature_vector import FeatureVector
from scoring.signal_models import Confidence, SignalType, TradeSignal


class DecisionEngine:
    """Generates trade signals from pump/dump scores and features.

    Applies threshold logic to classify signals and computes suggested
    entry, stop loss, and take profit levels.
    """

    def __init__(
        self, scoring_config: ScoringConfig, trading_config: Optional[TradingConfig] = None
    ) -> None:
        """Initialize with scoring thresholds and trading parameters.

        Args:
            scoring_config: Configuration with score thresholds.
            trading_config: Optional trading config for SL/TP multipliers.
        """
        self._scoring_config = scoring_config
        self._trading_config = trading_config or TradingConfig()

    def generate_signal(
        self,
        pump_score: float,
        dump_score: float,
        pump_reasons: List[str],
        dump_reasons: List[str],
        feature_vector: FeatureVector,
    ) -> TradeSignal:
        """Generate a trade signal from scoring outputs.

        Decision logic:
            - pump_score > 75 AND dump_score < 40: HIGH_PROBABILITY_LONG (HIGH)
            - pump_score 60-75 AND dump_score < 50: HIGH_PROBABILITY_LONG (MEDIUM)
            - pump_score 50-75: EARLY_ACCUMULATION_WATCH
            - dump_score > 70 AND pump_score < 40: HIGH_RISK_DUMP_SHORT (HIGH)
            - dump_score 60-70: HIGH_RISK_DUMP_SHORT (MEDIUM)
            - Else: NO_TRADE

        Args:
            pump_score: Normalized pump probability (0-100).
            dump_score: Normalized dump probability (0-100).
            pump_reasons: List of reasons contributing to pump score.
            dump_reasons: List of reasons contributing to dump score.
            feature_vector: Full feature vector for additional context.

        Returns:
            TradeSignal with classification, confidence, and trade parameters.
        """
        signal_type = SignalType.NO_TRADE
        confidence = Confidence.LOW
        reasons: List[str] = []

        # High probability long signals
        if pump_score > 75.0 and dump_score < 40.0:
            signal_type = SignalType.HIGH_PROBABILITY_LONG
            confidence = Confidence.HIGH
            reasons = pump_reasons[:]

        elif pump_score >= 60.0 and pump_score <= 75.0 and dump_score < 50.0:
            signal_type = SignalType.HIGH_PROBABILITY_LONG
            confidence = Confidence.MEDIUM
            reasons = pump_reasons[:]

        # Early accumulation watch
        elif pump_score >= 50.0 and pump_score <= 75.0:
            signal_type = SignalType.EARLY_ACCUMULATION_WATCH
            confidence = self._compute_confidence(
                pump_score, dump_score, SignalType.EARLY_ACCUMULATION_WATCH
            )
            reasons = pump_reasons[:]

        # High risk dump/short signals
        elif dump_score > 70.0 and pump_score < 40.0:
            signal_type = SignalType.HIGH_RISK_DUMP_SHORT
            confidence = Confidence.HIGH
            reasons = dump_reasons[:]

        elif dump_score >= 60.0 and dump_score <= 70.0:
            signal_type = SignalType.HIGH_RISK_DUMP_SHORT
            confidence = Confidence.MEDIUM
            reasons = dump_reasons[:]

        else:
            signal_type = SignalType.NO_TRADE
            confidence = Confidence.LOW
            reasons = []

        # Compute trade parameters for actionable signals
        entry_price: Optional[float] = None
        stop_loss: Optional[float] = None
        take_profit: Optional[float] = None

        if signal_type in (
            SignalType.HIGH_PROBABILITY_LONG,
            SignalType.HIGH_RISK_DUMP_SHORT,
        ):
            entry_price, stop_loss, take_profit = self._compute_trade_params(
                signal_type, feature_vector
            )

        return TradeSignal(
            symbol=feature_vector.symbol,
            signal_type=signal_type,
            pump_score=pump_score,
            dump_score=dump_score,
            confidence=confidence,
            reasons=reasons,
            timestamp=time.time(),
            entry_price=entry_price,
            suggested_stop_loss=stop_loss,
            suggested_take_profit=take_profit,
        )

    def _compute_confidence(
        self,
        pump_score: float,
        dump_score: float,
        signal_type: SignalType,
    ) -> Confidence:
        """Compute confidence level based on score margins.

        Args:
            pump_score: Normalized pump score.
            dump_score: Normalized dump score.
            signal_type: The signal type being evaluated.

        Returns:
            Confidence enum value.
        """
        if signal_type == SignalType.EARLY_ACCUMULATION_WATCH:
            margin = pump_score - dump_score
            if margin > 40.0:
                return Confidence.HIGH
            elif margin > 20.0:
                return Confidence.MEDIUM
            else:
                return Confidence.LOW

        if signal_type == SignalType.HIGH_PROBABILITY_LONG:
            if pump_score > 80.0 and dump_score < 30.0:
                return Confidence.HIGH
            elif pump_score > 65.0:
                return Confidence.MEDIUM
            return Confidence.LOW

        if signal_type == SignalType.HIGH_RISK_DUMP_SHORT:
            if dump_score > 80.0 and pump_score < 30.0:
                return Confidence.HIGH
            elif dump_score > 65.0:
                return Confidence.MEDIUM
            return Confidence.LOW

        return Confidence.LOW

    def _compute_trade_params(
        self,
        signal_type: SignalType,
        feature_vector: FeatureVector,
    ) -> tuple:
        """Compute entry, stop loss, and take profit levels.

        Uses ATR ratio for stop loss distance and applies R:R multiplier
        for take profit.

        Args:
            signal_type: Type of signal (long or short).
            feature_vector: Feature vector with ATR data.

        Returns:
            Tuple of (entry_price, stop_loss, take_profit) or (None, None, None).
        """
        # If no entry price context available, return None
        # Entry price would come from current market price in live trading
        # Here we compute relative SL/TP based on ATR
        atr_ratio = feature_vector.atr_ratio
        if atr_ratio <= 0.0:
            return None, None, None

        sl_multiplier = self._trading_config.stop_loss_atr_multiplier
        tp_multiplier = self._trading_config.take_profit_atr_multiplier

        # Entry price is not known here (comes from market at execution time)
        # We store the ATR-based distances as suggestions
        # The execution module will compute actual prices
        entry_price = None
        stop_loss_distance = atr_ratio * sl_multiplier
        take_profit_distance = atr_ratio * tp_multiplier

        if signal_type == SignalType.HIGH_PROBABILITY_LONG:
            # Long: SL below entry, TP above entry
            # Store as relative percentages for the executor
            stop_loss = -stop_loss_distance
            take_profit = take_profit_distance
        elif signal_type == SignalType.HIGH_RISK_DUMP_SHORT:
            # Short: SL above entry, TP below entry
            stop_loss = stop_loss_distance
            take_profit = -take_profit_distance
        else:
            return None, None, None

        return entry_price, stop_loss, take_profit

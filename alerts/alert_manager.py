"""
Alert manager - orchestrates all alert channels.

Dispatches signals to enabled alert channels (console, telegram, JSON feed)
and provides filtering for actionable signals only.
"""

from __future__ import annotations

import logging
from typing import Any, List

from alerts.console_dashboard import ConsoleDashboard
from alerts.json_feed import JSONFeed
from alerts.telegram_bot import TelegramAlertBot
from scoring.signal_models import SignalType, TradeSignal

logger = logging.getLogger(__name__)


class AlertManager:
    """Orchestrates signal dispatch to all enabled alert channels.

    Manages console dashboard, Telegram bot, and JSON feed outputs.
    Filters signals to only dispatch actionable ones (non-NO_TRADE).
    """

    def __init__(self, config: Any) -> None:
        """Initialize alert manager with all configured channels.

        Args:
            config: AlertConfig with channel enable/disable settings.
        """
        self._config = config

        # Initialize enabled channels
        self._console: ConsoleDashboard | None = None
        self._telegram: TelegramAlertBot | None = None
        self._json_feed: JSONFeed | None = None

        if getattr(config, "console_enabled", True):
            self._console = ConsoleDashboard(config)

        if getattr(config, "telegram_enabled", False):
            self._telegram = TelegramAlertBot(config)

        if getattr(config, "json_feed_enabled", True):
            output_dir = "output"
            self._json_feed = JSONFeed(config, output_dir=output_dir)

    async def dispatch_signal(self, signal: TradeSignal) -> None:
        """Send a single signal to all enabled channels.

        Args:
            signal: TradeSignal to dispatch.
        """
        # Skip NO_TRADE signals for individual dispatch
        if signal.signal_type == SignalType.NO_TRADE:
            return

        if self._telegram is not None:
            try:
                await self._telegram.send_signal_alert(signal)
            except Exception as e:
                logger.error("Telegram dispatch failed: %s", str(e))

        if self._json_feed is not None:
            try:
                self._json_feed.write_signal_history(signal)
            except Exception as e:
                logger.error("JSON feed history write failed: %s", str(e))

    async def dispatch_batch(self, signals: List[TradeSignal]) -> None:
        """Batch update to all enabled channels.

        Filters actionable signals and displays/writes them.

        Args:
            signals: Full list of signals from a scan cycle.
        """
        actionable = self._filter_actionable_signals(signals)

        # Console display (shows all actionable signals)
        if self._console is not None:
            try:
                self._console.display_signals(actionable)
            except Exception as e:
                logger.error("Console display failed: %s", str(e))

        # JSON feed (write current state)
        if self._json_feed is not None:
            try:
                self._json_feed.write_feed_file(actionable)
            except Exception as e:
                logger.error("JSON feed write failed: %s", str(e))

        # Telegram (send top signals only to avoid spam)
        if self._telegram is not None:
            for signal in actionable[:5]:
                try:
                    await self._telegram.send_signal_alert(signal)
                except Exception as e:
                    logger.error(
                        "Telegram batch dispatch failed for %s: %s",
                        signal.symbol, str(e),
                    )

    async def dispatch_risk_alert(self, message: str) -> None:
        """Send high-priority risk alert to all channels.

        Args:
            message: Risk alert message text.
        """
        logger.warning("RISK ALERT: %s", message)

        if self._console is not None:
            try:
                self._console._print(f"\n!!! RISK ALERT: {message} !!!\n")
            except Exception as e:
                logger.error("Console risk alert failed: %s", str(e))

        if self._telegram is not None:
            try:
                await self._telegram.send_risk_alert(message)
            except Exception as e:
                logger.error("Telegram risk alert failed: %s", str(e))

    def _filter_actionable_signals(
        self, signals: List[TradeSignal]
    ) -> List[TradeSignal]:
        """Filter out NO_TRADE signals for alerting.

        Args:
            signals: Full list of signals.

        Returns:
            List containing only actionable (non-NO_TRADE) signals.
        """
        return [s for s in signals if s.signal_type != SignalType.NO_TRADE]

"""
Telegram alert bot for sending trade signal notifications.

Uses raw aiohttp HTTP calls to the Telegram Bot API for simplicity
and minimal dependencies. Supports signal alerts, daily summaries,
and urgent risk notifications.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional

from scoring.signal_models import Confidence, SignalType, TradeSignal

logger = logging.getLogger(__name__)

# Telegram Bot API base URL
TELEGRAM_API_BASE = "https://api.telegram.org/bot{token}/sendMessage"


class TelegramAlertBot:
    """Sends formatted alerts to Telegram via the Bot API.

    Uses aiohttp for async HTTP requests directly to the Telegram API,
    avoiding the need for the python-telegram-bot library.
    """

    def __init__(self, config: Any) -> None:
        """Initialize Telegram bot with configuration.

        Args:
            config: AlertConfig containing bot_token and chat_id.
        """
        self._token = config.telegram_bot_token
        self._chat_id = config.telegram_chat_id
        self._enabled = config.telegram_enabled
        self._session: Any = None

    async def send_signal_alert(self, signal: TradeSignal) -> None:
        """Send a formatted signal alert to Telegram.

        Includes emoji indicators and all signal fields.

        Args:
            signal: TradeSignal to notify about.
        """
        if not self._enabled:
            return

        message = self._format_signal_message(signal)
        await self._send_message(message)

    async def send_daily_summary(
        self, signals: List[TradeSignal], daily_pnl: float
    ) -> None:
        """Send end-of-day summary report.

        Args:
            signals: All signals generated during the day.
            daily_pnl: Cumulative daily profit/loss.
        """
        if not self._enabled:
            return

        message = self._format_summary_message(signals, daily_pnl)
        await self._send_message(message)

    async def send_risk_alert(self, message: str) -> None:
        """Send urgent risk notification.

        Args:
            message: Risk alert message text.
        """
        if not self._enabled:
            return

        formatted = (
            "🚨 *RISK ALERT* 🚨\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            f"{message}\n"
            "━━━━━━━━━━━━━━━━━━━━\n"
            "⚠️ Immediate attention required"
        )
        await self._send_message(formatted)

    def _format_signal_message(self, signal: TradeSignal) -> str:
        """Build formatted Telegram message for a trade signal.

        Args:
            signal: TradeSignal to format.

        Returns:
            Formatted message string with emoji and markdown.
        """
        # Choose emoji based on signal type
        emoji_map = {
            SignalType.HIGH_PROBABILITY_LONG: "🚀",
            SignalType.EARLY_ACCUMULATION_WATCH: "👀",
            SignalType.HIGH_RISK_DUMP_SHORT: "💀",
            SignalType.NO_TRADE: "⏸️",
        }
        emoji = emoji_map.get(signal.signal_type, "📊")

        # Format confidence
        conf_emoji = {
            Confidence.HIGH: "🟢",
            Confidence.MEDIUM: "🟡",
            Confidence.LOW: "🔴",
        }
        conf_icon = conf_emoji.get(signal.confidence, "⚪")

        # Format reasons
        reasons_text = ""
        if signal.reasons:
            reasons_list = "\n".join(f"  - {r}" for r in signal.reasons[:5])
            reasons_text = f"\n📋 *Reasons:*\n{reasons_list}"

        # Format trade parameters if available
        trade_params = ""
        if signal.entry_price is not None:
            trade_params += f"\n💰 Entry: {signal.entry_price}"
        if signal.suggested_stop_loss is not None:
            trade_params += f"\n🛑 SL: {signal.suggested_stop_loss}"
        if signal.suggested_take_profit is not None:
            trade_params += f"\n🎯 TP: {signal.suggested_take_profit}"

        message = (
            f"{emoji} *EAGLE SIGNAL* {emoji}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"*Symbol:* `{signal.symbol}`\n"
            f"*Signal:* {signal.signal_type.value.replace('_', ' ')}\n"
            f"*Confidence:* {conf_icon} {signal.confidence.value}\n"
            f"\n"
            f"📈 *Pump Score:* {signal.pump_score:.1f}/100\n"
            f"📉 *Dump Score:* {signal.dump_score:.1f}/100\n"
            f"{reasons_text}"
            f"{trade_params}\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )

        return message

    def _format_summary_message(
        self, signals: List[TradeSignal], pnl: float
    ) -> str:
        """Build formatted daily summary message.

        Args:
            signals: All signals from the day.
            pnl: Daily PnL value.

        Returns:
            Formatted summary string.
        """
        total = len(signals)
        longs = sum(
            1 for s in signals
            if s.signal_type == SignalType.HIGH_PROBABILITY_LONG
        )
        shorts = sum(
            1 for s in signals
            if s.signal_type == SignalType.HIGH_RISK_DUMP_SHORT
        )
        watches = sum(
            1 for s in signals
            if s.signal_type == SignalType.EARLY_ACCUMULATION_WATCH
        )

        pnl_emoji = "🟢" if pnl >= 0 else "🔴"
        pnl_pct = pnl * 100

        message = (
            f"📊 *EAGLE Daily Summary* 📊\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"\n"
            f"*Signals Generated:* {total}\n"
            f"  🚀 Long: {longs}\n"
            f"  💀 Short: {shorts}\n"
            f"  👀 Watch: {watches}\n"
            f"\n"
            f"*Daily PnL:* {pnl_emoji} {pnl_pct:+.2f}%\n"
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )

        return message

    async def _send_message(self, text: str) -> None:
        """Send a message via the Telegram Bot API.

        Uses aiohttp to make an HTTP POST request. Handles errors
        gracefully so a Telegram failure does not crash the system.

        Args:
            text: Message text to send (supports Telegram Markdown).
        """
        if not self._token or not self._chat_id:
            logger.warning("Telegram bot token or chat_id not configured")
            return

        try:
            import aiohttp  # type: ignore[import]

            url = TELEGRAM_API_BASE.format(token=self._token)
            payload = {
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": "Markdown",
                "disable_web_page_preview": True,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=10) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        logger.warning(
                            "Telegram API returned %d: %s", resp.status, body
                        )
                    else:
                        logger.debug("Telegram message sent successfully")

        except ImportError:
            logger.warning(
                "aiohttp not installed - Telegram notifications disabled"
            )
        except Exception as e:
            logger.error("Failed to send Telegram message: %s", str(e))

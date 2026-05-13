"""
Console dashboard for displaying trading signals.

Provides formatted terminal output showing ranked signals, market overview,
and system status. Uses rich library when available, falls back to plain
text formatting otherwise.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List

from scoring.signal_models import Confidence, SignalType, TradeSignal


class ConsoleDashboard:
    """Console-based dashboard for displaying trade signals and system status.

    Attempts to use the rich library for colorful formatted output.
    Falls back to plain aligned text if rich is not installed.
    """

    def __init__(self, config: Any) -> None:
        """Initialize console dashboard.

        Args:
            config: AlertConfig containing display preferences.
        """
        self._config = config
        self._use_rich = False

        try:
            from rich.console import Console  # type: ignore[import]
            from rich.table import Table  # type: ignore[import]

            self._console = Console()
            self._use_rich = True
        except ImportError:
            self._console = None

    def display_signals(self, signals: List[TradeSignal]) -> None:
        """Display a formatted table of ranked trade signals.

        Columns: Rank, Symbol, Pump Score, Dump Score, Signal, Confidence, Top Reason.

        Args:
            signals: List of TradeSignal objects, already sorted by pump_score descending.
        """
        if not signals:
            self._print("No actionable signals detected.")
            return

        if self._use_rich:
            self._display_signals_rich(signals)
        else:
            self._display_signals_plain(signals)

    def display_market_overview(self, total_symbols: int, signals_by_type: Dict[str, int]) -> None:
        """Display summary of scan results.

        Args:
            total_symbols: Total number of symbols scanned.
            signals_by_type: Dictionary mapping signal type name to count.
        """
        self._print("")
        self._print("=" * 60)
        self._print("  MARKET OVERVIEW")
        self._print("=" * 60)
        self._print(f"  Total Symbols Scanned: {total_symbols}")
        self._print("")

        for signal_type, count in signals_by_type.items():
            self._print(f"  {signal_type}: {count}")

        self._print("=" * 60)
        self._print("")

    def display_startup_banner(self) -> None:
        """Display ASCII art startup banner for the EAGLE system."""
        banner = r"""
    ███████╗ █████╗  ██████╗ ██╗     ███████╗
    ██╔════╝██╔══██╗██╔════╝ ██║     ██╔════╝
    █████╗  ███████║██║  ███╗██║     █████╗
    ██╔══╝  ██╔══██║██║   ██║██║     ██╔══╝
    ███████╗██║  ██║╚██████╔╝███████╗███████╗
    ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚══════╝

    Crypto Futures Intelligence System
    -----------------------------------
    Mode: Real-time Pump/Dump Detection
    Version: 1.0.0
"""
        self._print(banner)

    def _display_signals_rich(self, signals: List[TradeSignal]) -> None:
        """Display signals using rich library formatting."""
        try:
            from rich.table import Table  # type: ignore[import]

            table = Table(title="EAGLE Signal Dashboard")
            table.add_column("Rank", justify="center", style="bold")
            table.add_column("Symbol", style="cyan")
            table.add_column("Pump", justify="right", style="green")
            table.add_column("Dump", justify="right", style="red")
            table.add_column("Signal", style="bold")
            table.add_column("Confidence", justify="center")
            table.add_column("Top Reason")

            for rank, signal in enumerate(signals, 1):
                row = self._format_signal_row(signal, rank)
                pump_style = "bold green" if signal.pump_score > 70 else "green"
                dump_style = "bold red" if signal.dump_score > 70 else "red"
                table.add_row(
                    str(rank),
                    row["symbol"],
                    row["pump_score"],
                    row["dump_score"],
                    row["signal"],
                    row["confidence"],
                    row["reason"],
                )

            self._console.print(table)  # type: ignore[union-attr]
        except Exception:
            self._display_signals_plain(signals)

    def _display_signals_plain(self, signals: List[TradeSignal]) -> None:
        """Display signals using plain text formatting."""
        header = (
            f"{'Rank':<5} {'Symbol':<15} {'Pump':<7} {'Dump':<7} "
            f"{'Signal':<28} {'Conf':<8} {'Top Reason'}"
        )
        separator = "-" * 100

        self._print("")
        self._print("  EAGLE Signal Dashboard")
        self._print(separator)
        self._print(header)
        self._print(separator)

        for rank, signal in enumerate(signals, 1):
            row = self._format_signal_row(signal, rank)
            line = (
                f"{rank:<5} {row['symbol']:<15} "
                f"{self._color_score(signal.pump_score, 'pump')}{row['pump_score']:<7}\033[0m "
                f"{self._color_score(signal.dump_score, 'dump')}{row['dump_score']:<7}\033[0m "
                f"{row['signal']:<28} {row['confidence']:<8} {row['reason']}"
            )
            self._print(line)

        self._print(separator)
        self._print(f"  Total signals: {len(signals)}")
        self._print("")

    def _format_signal_row(self, signal: TradeSignal, rank: int) -> Dict[str, str]:
        """Format a single signal into display fields.

        Args:
            signal: TradeSignal to format.
            rank: Rank position in the list.

        Returns:
            Dictionary with formatted string fields.
        """
        reason = signal.reasons[0] if signal.reasons else "N/A"
        # Truncate long reasons
        if len(reason) > 50:
            reason = reason[:47] + "..."

        signal_name = signal.signal_type.value.replace("_", " ")

        return {
            "rank": str(rank),
            "symbol": signal.symbol,
            "pump_score": f"{signal.pump_score:.1f}",
            "dump_score": f"{signal.dump_score:.1f}",
            "signal": signal_name,
            "confidence": signal.confidence.value,
            "reason": reason,
        }

    def _color_score(self, score: float, score_type: str = "pump") -> str:
        """Return ANSI color code based on score magnitude.

        Args:
            score: Score value (0-100).
            score_type: Either 'pump' (green shades) or 'dump' (red shades).

        Returns:
            ANSI escape code string for the appropriate color.
        """
        if score_type == "pump":
            if score > 75:
                return "\033[92m"  # Bright green
            elif score > 50:
                return "\033[32m"  # Green
            else:
                return "\033[37m"  # White/dim
        else:
            if score > 75:
                return "\033[91m"  # Bright red
            elif score > 50:
                return "\033[31m"  # Red
            else:
                return "\033[37m"  # White/dim

    def _print(self, text: str) -> None:
        """Print text to stdout.

        Args:
            text: Text to print.
        """
        print(text, flush=True)

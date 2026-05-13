"""
JSON feed output for trade signals.

Writes signals to JSON files for consumption by external systems,
APIs, or dashboards. Supports both current-state feed files and
append-only history logs.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List

from scoring.signal_models import TradeSignal


class JSONFeed:
    """Manages JSON output of trade signals for external consumption.

    Writes current signals to a feed file and appends historical
    signals to a JSONL history file.
    """

    def __init__(self, config: Any, output_dir: str = "output") -> None:
        """Initialize JSON feed writer.

        Args:
            config: AlertConfig containing feed path preferences.
            output_dir: Directory for output files. Created if not exists.
        """
        self._config = config
        self._output_dir = output_dir

    def generate_signal_json(self, signal: TradeSignal) -> Dict[str, Any]:
        """Generate a signal dictionary matching the spec format.

        Output format:
            {
                "symbol": "...",
                "pump_score": 82,
                "dump_score": 18,
                "signal": "HIGH PROBABILITY LONG SETUP",
                "confidence": "HIGH",
                "reason": [...]
            }

        Args:
            signal: TradeSignal to convert.

        Returns:
            Dictionary in the exact spec format.
        """
        # Map signal type to human-readable description
        signal_descriptions = {
            "HIGH_PROBABILITY_LONG": "HIGH PROBABILITY LONG SETUP",
            "EARLY_ACCUMULATION_WATCH": "EARLY ACCUMULATION WATCH",
            "HIGH_RISK_DUMP_SHORT": "HIGH RISK DUMP SHORT",
            "NO_TRADE": "NO TRADE",
        }

        signal_text = signal_descriptions.get(
            signal.signal_type.value, signal.signal_type.value
        )

        return {
            "symbol": signal.symbol,
            "pump_score": int(round(signal.pump_score)),
            "dump_score": int(round(signal.dump_score)),
            "signal": signal_text,
            "confidence": signal.confidence.value,
            "reason": list(signal.reasons),
        }

    def write_feed_file(self, signals: List[TradeSignal]) -> None:
        """Write current signals to the feed JSON file.

        Creates the output directory if it does not exist.
        Overwrites the file with the latest state.

        Args:
            signals: List of current trade signals to write.
        """
        self._ensure_output_dir()

        feed = self.get_current_feed(signals)
        filepath = os.path.join(self._output_dir, "signals_feed.json")

        try:
            with open(filepath, "w") as f:
                json.dump(feed, f, indent=2)
        except OSError as e:
            import logging
            logging.getLogger(__name__).error(
                "Failed to write feed file: %s", str(e)
            )

    def write_signal_history(self, signal: TradeSignal) -> None:
        """Append a single signal to the history JSONL file.

        Each line is a self-contained JSON object for easy streaming reads.

        Args:
            signal: TradeSignal to append to history.
        """
        self._ensure_output_dir()

        filepath = os.path.join(self._output_dir, "signal_history.jsonl")
        entry = self.generate_signal_json(signal)
        entry["timestamp"] = signal.timestamp

        try:
            with open(filepath, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError as e:
            import logging
            logging.getLogger(__name__).error(
                "Failed to write signal history: %s", str(e)
            )

    def get_current_feed(self, signals: List[TradeSignal]) -> Dict[str, Any]:
        """Return the full feed dictionary with metadata.

        Includes scan timestamp, total symbols context, active signal count,
        and the signal list.

        Args:
            signals: List of current trade signals.

        Returns:
            Complete feed dictionary with metadata and signals.
        """
        signal_dicts = [self.generate_signal_json(s) for s in signals]
        active_count = sum(
            1 for s in signals if s.signal_type.value != "NO_TRADE"
        )

        return {
            "scan_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "total_symbols": len(signals),
            "active_signals": active_count,
            "signals": signal_dicts,
        }

    def _ensure_output_dir(self) -> None:
        """Create the output directory if it does not exist."""
        os.makedirs(self._output_dir, exist_ok=True)

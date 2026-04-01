"""Pipeline timing and performance profiling."""

import logging
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """Profiles pipeline stages to identify bottlenecks and produce waterfall charts."""

    def __init__(self) -> None:
        """Initialize the profiler."""
        self._stages: dict[str, dict[str, Any]] = {}
        self._stage_order: list[str] = []

    def start_stage(self, stage_name: str) -> None:
        """Record the start time of a pipeline stage.

        Args:
            stage_name: Name of the stage being started.
        """
        self._stages[stage_name] = {
            "start": time.time(),
            "end": None,
            "duration_ms": None,
        }
        if stage_name not in self._stage_order:
            self._stage_order.append(stage_name)
        logger.debug("Stage started: %s", stage_name)

    def end_stage(self, stage_name: str) -> None:
        """Record the end time of a pipeline stage.

        Args:
            stage_name: Name of the stage being completed.
        """
        if stage_name not in self._stages:
            logger.warning("end_stage called for unknown stage: %s", stage_name)
            return

        end_time = time.time()
        stage = self._stages[stage_name]
        stage["end"] = end_time
        stage["duration_ms"] = (end_time - stage["start"]) * 1000.0
        logger.debug(
            "Stage completed: %s (%.1fms)", stage_name, stage["duration_ms"]
        )

    def get_profile(self) -> dict[str, Any]:
        """Get the complete profiling results.

        Returns:
            Dict with stages list, total_ms, and slowest_stage.
        """
        stages: list[dict[str, Any]] = []
        total_ms = 0.0
        slowest_name: Optional[str] = None
        slowest_ms = 0.0

        for name in self._stage_order:
            info = self._stages.get(name, {})
            duration = info.get("duration_ms")
            if duration is not None:
                stages.append({
                    "name": name,
                    "duration_ms": round(duration, 1),
                })
                total_ms += duration
                if duration > slowest_ms:
                    slowest_ms = duration
                    slowest_name = name

        return {
            "stages": stages,
            "total_ms": round(total_ms, 1),
            "slowest_stage": slowest_name,
        }

    def format_waterfall(self) -> str:
        """Generate a text-based waterfall chart of stage timings.

        Returns:
            Multi-line string with a visual bar chart of stage durations.
        """
        profile = self.get_profile()
        stages = profile["stages"]
        total_ms = profile["total_ms"]

        if not stages:
            return "No stages recorded."

        # Calculate display parameters
        max_name_len = max(len(s["name"]) for s in stages)
        bar_width = 10

        lines: list[str] = []
        for stage in stages:
            name = stage["name"].ljust(max_name_len)
            duration_ms = stage["duration_ms"]
            duration_s = duration_ms / 1000.0

            # Calculate filled portion of bar
            if total_ms > 0:
                fill = int((duration_ms / total_ms) * bar_width)
            else:
                fill = 0
            fill = max(1, min(fill, bar_width))
            empty = bar_width - fill

            bar = "\u2588" * fill + "\u2591" * empty
            lines.append(f"{name}: {bar} {duration_s:.1f}s")

        # Total line
        total_s = total_ms / 1000.0
        total_label = "Total".ljust(max_name_len)
        lines.append(f"{total_label}:            {total_s:.1f}s")

        return "\n".join(lines)

    def get_bottleneck(self) -> dict[str, Any]:
        """Identify the slowest pipeline stage.

        Returns:
            Dict with stage name, duration_ms, and percentage_of_total.
        """
        profile = self.get_profile()
        stages = profile["stages"]
        total_ms = profile["total_ms"]

        if not stages:
            return {"stage": None, "duration_ms": 0, "percentage_of_total": 0}

        slowest = max(stages, key=lambda s: s["duration_ms"])
        pct = (slowest["duration_ms"] / total_ms * 100) if total_ms > 0 else 0

        return {
            "stage": slowest["name"],
            "duration_ms": round(slowest["duration_ms"], 1),
            "percentage_of_total": round(pct, 1),
        }

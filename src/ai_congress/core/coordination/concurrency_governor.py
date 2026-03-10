"""GPU-aware dynamic concurrency control for model parallelism."""

import asyncio
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = {0.8: 1, 0.5: 3, 0.0: 5}


class ConcurrencyGovernor:
    """Dynamically limits model query parallelism based on GPU VRAM usage.

    Polls nvidia-smi at a configurable interval and adjusts the concurrency
    limit. Uses an asyncio.Condition with explicit counter tracking so that
    limit changes take effect immediately without creating fresh permits.
    """

    def __init__(
        self,
        poll_interval: float = 2.0,
        vram_thresholds: dict[float, int] | None = None,
        min_concurrent: int = 1,
        max_concurrent: int = 5,
        gpu_index: int = 0,
    ):
        self.poll_interval = poll_interval
        self.vram_thresholds = vram_thresholds or dict(DEFAULT_THRESHOLDS)
        self.min_concurrent = min_concurrent
        self.max_concurrent = max_concurrent
        self.gpu_index = gpu_index

        self._dynamic_limit = max_concurrent
        self._cond = asyncio.Condition()
        self._poll_task: asyncio.Task | None = None
        self._running = False

        # Stats
        self._vram_used: int = 0
        self._vram_total: int = 1  # avoid division by zero
        self._gpu_util: int = 0
        self._current_usage: float = 0.0
        self._active_count: int = 0
        self._waiting_count: int = 0

    def _calculate_limit(self, usage_ratio: float) -> int:
        """Map VRAM usage ratio to concurrency limit using thresholds."""
        for threshold in sorted(self.vram_thresholds.keys(), reverse=True):
            if usage_ratio >= threshold:
                return self.vram_thresholds[threshold]
        return self.max_concurrent

    def _parse_nvidia_smi(self, output: str) -> tuple[int, int, int]:
        """Parse nvidia-smi CSV output into (used_mb, total_mb, util_pct)."""
        try:
            parts = [p.strip() for p in output.strip().split(",")]
            return int(parts[0]), int(parts[1]), int(parts[2])
        except (ValueError, IndexError):
            logger.warning("Failed to parse nvidia-smi output: %s", output.strip())
            return 0, 1, 0

    async def _poll_gpu_once(self) -> tuple[int, int, int]:
        """Run nvidia-smi and return (used_mb, total_mb, util_pct)."""
        try:
            proc = await asyncio.create_subprocess_exec(
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
                "-i", str(self.gpu_index),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
            return self._parse_nvidia_smi(stdout.decode())
        except FileNotFoundError:
            logger.warning("nvidia-smi not found — defaulting to max concurrency")
            return 0, 1, 0
        except Exception as e:
            logger.warning("GPU poll failed: %s", e)
            return 0, 1, 0

    async def _poll_loop(self):
        """Background loop that polls GPU state and adjusts the concurrency limit."""
        while self._running:
            try:
                used, total, util = await self._poll_gpu_once()
                self._vram_used = used
                self._vram_total = total if total > 0 else 1
                self._gpu_util = util
                self._current_usage = used / self._vram_total

                new_limit = self._calculate_limit(self._current_usage)
                new_limit = max(self.min_concurrent, min(self.max_concurrent, new_limit))

                if new_limit != self._dynamic_limit:
                    old_limit = self._dynamic_limit
                    self._dynamic_limit = new_limit
                    logger.info(
                        "GPU concurrency limit changed: %d -> %d (VRAM: %.0f%%)",
                        old_limit, new_limit, self._current_usage * 100,
                    )
                    # Wake waiters — some may now fit under the new (higher) limit
                    if new_limit > old_limit:
                        async with self._cond:
                            self._cond.notify_all()
            except Exception as e:
                logger.warning("GPU poll loop error: %s", e)

            await asyncio.sleep(self.poll_interval)

    async def start(self):
        """Start background GPU polling."""
        if self._running:
            return
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("ConcurrencyGovernor started (poll_interval=%.1fs)", self.poll_interval)

    async def stop(self):
        """Stop background GPU polling and wake all waiters."""
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None
        # Wake any blocked waiters so they don't hang after shutdown
        async with self._cond:
            self._dynamic_limit = self.max_concurrent
            self._cond.notify_all()
        logger.info("ConcurrencyGovernor stopped")

    async def acquire(self):
        """Acquire a concurrency slot. Blocks if at the dynamic limit."""
        self._waiting_count += 1
        try:
            async with self._cond:
                while self._active_count >= self._dynamic_limit:
                    await self._cond.wait()
                self._active_count += 1
        except BaseException:
            self._waiting_count -= 1
            raise
        self._waiting_count -= 1

    async def release(self):
        """Release a concurrency slot and wake one waiter."""
        async with self._cond:
            self._active_count = max(0, self._active_count - 1)
            self._cond.notify()

    @asynccontextmanager
    async def throttled(self):
        """Context manager for GPU-throttled execution."""
        await self.acquire()
        try:
            yield self.get_stats()
        finally:
            await self.release()

    async def __aenter__(self):
        """Start governor as async context manager."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Stop governor on context exit, ensuring cleanup on exceptions."""
        await self.stop()
        return False

    def get_stats(self) -> dict:
        """Return current GPU and concurrency stats."""
        return {
            "vram_used_mb": self._vram_used,
            "vram_total_mb": self._vram_total,
            "vram_usage_pct": self._current_usage * 100,
            "gpu_util_pct": self._gpu_util,
            "current_limit": self._dynamic_limit,
            "active_tasks": self._active_count,
            "waiting_tasks": self._waiting_count,
        }

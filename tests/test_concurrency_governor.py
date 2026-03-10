"""Tests for ConcurrencyGovernor — GPU-aware dynamic concurrency control."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch

from src.ai_congress.core.coordination.concurrency_governor import ConcurrencyGovernor


class TestConcurrencyGovernor:
    """Unit tests for ConcurrencyGovernor."""

    def test_default_thresholds(self):
        gov = ConcurrencyGovernor()
        assert gov.min_concurrent == 1
        assert gov.max_concurrent == 5
        assert gov.poll_interval == 2.0

    def test_calculate_limit_low_usage(self):
        gov = ConcurrencyGovernor()
        assert gov._calculate_limit(0.3) == 5

    def test_calculate_limit_medium_usage(self):
        gov = ConcurrencyGovernor()
        assert gov._calculate_limit(0.6) == 3

    def test_calculate_limit_high_usage(self):
        gov = ConcurrencyGovernor()
        assert gov._calculate_limit(0.85) == 1

    def test_calculate_limit_custom_thresholds(self):
        gov = ConcurrencyGovernor(
            vram_thresholds={0.9: 1, 0.7: 2, 0.0: 4}
        )
        assert gov._calculate_limit(0.95) == 1
        assert gov._calculate_limit(0.75) == 2
        assert gov._calculate_limit(0.5) == 4

    def test_parse_nvidia_smi_output(self):
        gov = ConcurrencyGovernor()
        used, total, util = gov._parse_nvidia_smi("8192, 23028, 45\n")
        assert used == 8192
        assert total == 23028
        assert util == 45

    def test_parse_nvidia_smi_malformed(self):
        gov = ConcurrencyGovernor()
        used, total, util = gov._parse_nvidia_smi("garbage")
        assert used == 0
        assert total == 1
        assert util == 0

    def test_get_stats_initial(self):
        gov = ConcurrencyGovernor()
        stats = gov.get_stats()
        assert stats["vram_used_mb"] == 0
        assert stats["vram_total_mb"] == 1
        assert stats["vram_usage_pct"] == 0.0
        assert stats["gpu_util_pct"] == 0
        assert stats["current_limit"] == gov.max_concurrent
        assert stats["active_tasks"] == 0
        assert stats["waiting_tasks"] == 0

    @pytest.mark.asyncio
    async def test_throttled_context_manager(self):
        gov = ConcurrencyGovernor()
        gov._dynamic_limit = 2

        acquired = []

        async def do_work(task_id):
            async with gov.throttled() as stats:
                acquired.append(task_id)
                assert stats["current_limit"] == 2
                await asyncio.sleep(0.01)

        await asyncio.gather(do_work(1), do_work(2))
        assert len(acquired) == 2

    @pytest.mark.asyncio
    async def test_concurrency_actually_limits(self):
        gov = ConcurrencyGovernor()
        gov._dynamic_limit = 1

        max_concurrent_seen = 0
        current_concurrent = 0

        async def do_work():
            nonlocal max_concurrent_seen, current_concurrent
            async with gov.throttled():
                current_concurrent += 1
                max_concurrent_seen = max(max_concurrent_seen, current_concurrent)
                await asyncio.sleep(0.05)
                current_concurrent -= 1

        await asyncio.gather(do_work(), do_work(), do_work())
        assert max_concurrent_seen == 1

    @pytest.mark.asyncio
    async def test_limit_change_blocks_new_arrivals(self):
        """When the limit drops mid-flight, new tasks must wait for active ones to finish."""
        gov = ConcurrencyGovernor()
        gov._dynamic_limit = 3

        events = []

        async def long_task(task_id):
            async with gov.throttled():
                events.append(f"start-{task_id}")
                await asyncio.sleep(0.1)
                events.append(f"end-{task_id}")

        async def late_task():
            # Wait a bit, then reduce limit and try to acquire
            await asyncio.sleep(0.02)
            gov._dynamic_limit = 1  # drop limit while 2 tasks are active
            async with gov.throttled():
                events.append("start-late")
                events.append(f"active_when_late={gov._active_count}")

        await asyncio.gather(long_task("a"), long_task("b"), late_task())

        # Late task must have started after at least one long task ended
        late_idx = events.index("start-late")
        ends_before_late = [e for e in events[:late_idx] if e.startswith("end-")]
        assert len(ends_before_late) >= 1, f"Late task started too early: {events}"

    @pytest.mark.asyncio
    async def test_start_stop_lifecycle(self):
        with patch.object(ConcurrencyGovernor, '_poll_gpu_once', new_callable=AsyncMock) as mock_poll:
            mock_poll.return_value = (4000, 23028, 30)
            gov = ConcurrencyGovernor(poll_interval=0.05)
            await gov.start()
            await asyncio.sleep(0.1)
            await gov.stop()
            assert mock_poll.call_count >= 1

    @pytest.mark.asyncio
    async def test_throttled_releases_on_exception(self):
        """If body of throttled() raises, the slot must still be released."""
        gov = ConcurrencyGovernor()
        gov._dynamic_limit = 1

        with pytest.raises(ValueError, match="boom"):
            async with gov.throttled():
                raise ValueError("boom")

        # Slot was released — active count back to 0
        assert gov._active_count == 0
        # Can acquire again (not permanently blocked)
        async with gov.throttled():
            assert gov._active_count == 1

    @pytest.mark.asyncio
    async def test_stop_wakes_blocked_waiters(self):
        """Calling stop() must unblock coroutines waiting in acquire()."""
        gov = ConcurrencyGovernor()
        gov._dynamic_limit = 1

        # Occupy the single slot
        await gov.acquire()
        assert gov._active_count == 1

        unblocked = asyncio.Event()

        async def blocked_acquirer():
            await gov.acquire()
            unblocked.set()

        task = asyncio.create_task(blocked_acquirer())
        await asyncio.sleep(0.01)  # let it block

        # stop() should wake the blocked acquirer
        await gov.stop()
        await asyncio.wait_for(unblocked.wait(), timeout=1.0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        """Governor supports async with for start/stop lifecycle."""
        gov = ConcurrencyGovernor()
        async with gov:
            assert gov._running is True
        # After exit, governor is stopped
        assert gov._running is False

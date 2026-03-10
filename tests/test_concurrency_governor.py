"""Tests for ConcurrencyGovernor — GPU-aware dynamic concurrency control."""

import asyncio
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

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
        gov._semaphore = asyncio.Semaphore(2)
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
        gov._semaphore = asyncio.Semaphore(1)
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
    async def test_start_stop_lifecycle(self):
        with patch.object(ConcurrencyGovernor, '_poll_gpu_once', new_callable=AsyncMock) as mock_poll:
            mock_poll.return_value = (4000, 23028, 30)
            gov = ConcurrencyGovernor(poll_interval=0.05)
            await gov.start()
            await asyncio.sleep(0.1)
            await gov.stop()
            assert mock_poll.call_count >= 1

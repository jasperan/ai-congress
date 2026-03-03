"""
Tests for ACP Wave Controller
"""
import pytest
import pytest_asyncio
from src.ai_congress.core.acp.wave_controller import Task, WaveController, WaveResult


class MockTask(Task):
    """Test task that returns a configurable result."""
    def __init__(self, task_id: str, result: str = "ok"):
        super().__init__(task_id=task_id)
        self._result = result

    async def execute(self) -> str:
        return self._result


class TestWaveController:
    @pytest.mark.asyncio
    async def test_single_wave_no_deps(self):
        controller = WaveController()
        tasks = [MockTask("a"), MockTask("b"), MockTask("c")]
        waves = []
        async for wave_result in controller.execute_waves(tasks):
            waves.append(wave_result)
        # All tasks should execute in a single wave (no dependencies)
        assert len(waves) == 1
        assert waves[0].wave == 1
        assert len(waves[0].tasks) == 3

    @pytest.mark.asyncio
    async def test_dependency_ordering(self):
        controller = WaveController()
        tasks = [MockTask("a"), MockTask("b"), MockTask("c")]
        # b depends on a, c depends on b
        controller.add_dependency("b", "a")
        controller.add_dependency("c", "b")
        waves = []
        async for wave_result in controller.execute_waves(tasks):
            waves.append(wave_result)
        assert len(waves) == 3
        assert waves[0].tasks[0].task_id == "a"
        assert waves[1].tasks[0].task_id == "b"
        assert waves[2].tasks[0].task_id == "c"

    @pytest.mark.asyncio
    async def test_parallel_independent_tasks(self):
        controller = WaveController()
        tasks = [MockTask("a"), MockTask("b"), MockTask("c"), MockTask("d")]
        # b and c depend on a, d depends on nothing
        controller.add_dependency("b", "a")
        controller.add_dependency("c", "a")
        waves = []
        async for wave_result in controller.execute_waves(tasks):
            waves.append(wave_result)
        # Wave 1: a and d (both have no unmet dependencies)
        # Wave 2: b and c (both depend on a, which is now completed)
        assert len(waves) == 2
        wave1_ids = {t.task_id for t in waves[0].tasks}
        wave2_ids = {t.task_id for t in waves[1].tasks}
        assert "a" in wave1_ids
        assert "d" in wave1_ids
        assert "b" in wave2_ids
        assert "c" in wave2_ids

    @pytest.mark.asyncio
    async def test_max_waves_limit(self):
        controller = WaveController(max_waves=2)
        tasks = [MockTask("a"), MockTask("b"), MockTask("c")]
        controller.add_dependency("b", "a")
        controller.add_dependency("c", "b")
        waves = []
        async for wave_result in controller.execute_waves(tasks):
            waves.append(wave_result)
        # Only 2 waves should execute due to max_waves limit
        assert len(waves) == 2
        # c should not have been executed
        executed_ids = set()
        for w in waves:
            for t in w.tasks:
                executed_ids.add(t.task_id)
        assert "a" in executed_ids
        assert "b" in executed_ids
        assert "c" not in executed_ids

    @pytest.mark.asyncio
    async def test_wave_results_contain_outputs(self):
        controller = WaveController()
        tasks = [MockTask("a", result="result_a"), MockTask("b", result="result_b")]
        waves = []
        async for wave_result in controller.execute_waves(tasks):
            waves.append(wave_result)
        assert len(waves) == 1
        assert "result_a" in waves[0].results
        assert "result_b" in waves[0].results

    @pytest.mark.asyncio
    async def test_empty_tasks(self):
        controller = WaveController()
        waves = []
        async for wave_result in controller.execute_waves([]):
            waves.append(wave_result)
        assert len(waves) == 0

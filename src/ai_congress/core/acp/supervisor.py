"""
AgentSupervisor - OTP-inspired supervision for agent tasks.

Provides fault-tolerant agent lifecycle management with:
- Restart-on-failure with error context injection
- Stall detection with configurable timeouts
- Exponential backoff retries
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

logger = logging.getLogger(__name__)


class RestartPolicy(str, Enum):
    RESTART = "restart"       # Restart with error context
    SKIP = "skip"             # Skip and mark failed
    FALLBACK = "fallback"     # Use fallback value


@dataclass
class SupervisedTask:
    agent_id: str
    coro_factory: Callable[..., Coroutine]
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    restart_policy: RestartPolicy = RestartPolicy.RESTART
    max_retries: int = 3
    stall_timeout: float = 120.0
    backoff_base: float = 1.0
    result: Any = None
    error: Optional[str] = None
    attempts: int = 0
    started_at: float = 0.0
    completed: bool = False
    success: bool = False


class AgentSupervisor:
    """OTP-inspired supervisor for concurrent agent tasks."""

    def __init__(
        self,
        max_retries: int = 3,
        stall_timeout: float = 120.0,
        backoff_base: float = 1.0,
    ):
        self.max_retries = max_retries
        self.stall_timeout = stall_timeout
        self.backoff_base = backoff_base
        self._tasks: dict[str, SupervisedTask] = {}
        self._results: dict[str, Any] = {}
        self._event_log: list[dict] = []

    def _log_event(self, agent_id: str, event: str, detail: str = ""):
        entry = {
            "timestamp": time.time(),
            "agent_id": agent_id,
            "event": event,
            "detail": detail,
        }
        self._event_log.append(entry)
        logger.info(f"[Supervisor] {agent_id}: {event} {detail}")

    async def _execute_with_timeout(self, task: SupervisedTask) -> Any:
        """Execute a task with stall detection timeout."""
        return await asyncio.wait_for(
            task.coro_factory(*task.args, **task.kwargs),
            timeout=task.stall_timeout,
        )

    async def _run_supervised(self, task: SupervisedTask) -> Any:
        """Run a single task with retry and backoff."""
        last_error = None
        for attempt in range(task.max_retries):
            task.attempts = attempt + 1
            task.started_at = time.time()

            try:
                self._log_event(task.agent_id, "START", f"attempt={attempt + 1}")
                result = await self._execute_with_timeout(task)
                task.result = result
                task.completed = True
                task.success = True
                self._log_event(task.agent_id, "COMPLETE", f"attempt={attempt + 1}")
                return result

            except asyncio.TimeoutError:
                last_error = f"Stalled after {task.stall_timeout}s"
                self._log_event(task.agent_id, "STALL", last_error)

                if task.restart_policy == RestartPolicy.SKIP:
                    break

                # Inject error context for next attempt
                if "error_context" not in task.kwargs:
                    task.kwargs["error_context"] = ""
                task.kwargs["error_context"] += f"\n[Previous attempt timed out after {task.stall_timeout}s. Please respond more concisely.]"

            except Exception as e:
                last_error = str(e)
                self._log_event(task.agent_id, "FAIL", f"attempt={attempt + 1}: {last_error}")

                if task.restart_policy == RestartPolicy.SKIP:
                    break

                if "error_context" not in task.kwargs:
                    task.kwargs["error_context"] = ""
                task.kwargs["error_context"] += f"\n[Previous attempt failed: {last_error}. Please try again.]"

            # Exponential backoff
            if attempt < task.max_retries - 1:
                delay = task.backoff_base * (2 ** attempt)
                self._log_event(task.agent_id, "BACKOFF", f"{delay:.1f}s")
                await asyncio.sleep(delay)

        # All retries exhausted
        task.completed = True
        task.success = False
        task.error = last_error
        self._log_event(task.agent_id, "EXHAUSTED", f"after {task.attempts} attempts")
        return None

    async def supervise_all(
        self,
        tasks: list[SupervisedTask],
    ) -> list[SupervisedTask]:
        """Run all tasks concurrently with supervision.

        Returns list of SupervisedTask with results populated.
        """
        self._tasks = {t.agent_id: t for t in tasks}

        coros = [self._run_supervised(t) for t in tasks]
        await asyncio.gather(*coros, return_exceptions=True)

        return tasks

    def get_event_log(self) -> list[dict]:
        return list(self._event_log)

    def get_summary(self) -> dict:
        """Return summary of supervised execution."""
        tasks = list(self._tasks.values())
        succeeded = [t for t in tasks if t.success]
        failed = [t for t in tasks if t.completed and not t.success]
        total_attempts = sum(t.attempts for t in tasks)
        return {
            "total_tasks": len(tasks),
            "succeeded": len(succeeded),
            "failed": len(failed),
            "total_attempts": total_attempts,
            "failed_agents": [t.agent_id for t in failed],
            "event_count": len(self._event_log),
        }

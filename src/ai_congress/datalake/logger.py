"""
EventLogger — non-blocking, fire-and-forget event logging to Oracle.

Events are queued in-memory and flushed in batches. If Oracle is unavailable,
events are silently dropped with a warning log. Never crashes the app.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .connection import OraclePoolManager

logger = logging.getLogger(__name__)


@dataclass
class Event:
    event_type: str
    session_id: str = ""
    event_data: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)


class EventLogger:
    """Non-blocking event logger that batches writes to Oracle."""

    def __init__(
        self,
        pool_manager: OraclePoolManager,
        batch_size: int = 20,
        flush_interval: float = 5.0,
    ):
        self._pool = pool_manager
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=10000)
        self._flush_task: Optional[asyncio.Task] = None
        self._sequence_counters: Dict[str, int] = {}
        self._running = False

    def start(self) -> None:
        """Start the background flush loop."""
        if self._running:
            return
        self._running = True
        self._flush_task = asyncio.create_task(self._flush_loop())
        logger.info("EventLogger started")

    async def stop(self) -> None:
        """Flush remaining events and stop."""
        self._running = False
        if self._flush_task:
            self._flush_task.cancel()
            try:
                await self._flush_task
            except asyncio.CancelledError:
                pass
        # Final flush
        await self._flush_batch()
        logger.info("EventLogger stopped")

    def _next_seq(self, session_id: str) -> int:
        """Get next sequence number for a session."""
        seq = self._sequence_counters.get(session_id, 0) + 1
        self._sequence_counters[session_id] = seq
        return seq

    def log(self, event_type: str, session_id: str = "", **data) -> None:
        """Queue an event for logging. Non-blocking, never raises."""
        try:
            event = Event(
                event_type=event_type,
                session_id=session_id,
                event_data=data,
            )
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Event queue full, dropping event: %s", event_type)
        except Exception as e:
            logger.warning("Failed to queue event %s: %s", event_type, e)

    def new_session(self) -> str:
        """Generate a new session ID."""
        return str(uuid.uuid4())

    async def log_session(
        self,
        session_id: str,
        prompt: str,
        mode: str,
        voting_mode: str,
        models: List[str],
    ) -> None:
        """Log a new session directly (not queued — small, important)."""
        if not self._pool.is_available:
            return
        try:
            async with self._pool.pool.acquire() as conn:
                cursor = conn.cursor()
                await cursor.execute(
                    """
                    INSERT INTO CONGRESS_SESSIONS
                        (session_id, prompt, swarm_mode, voting_mode, model_count, models_used)
                    VALUES (:sid, :prompt, :smode, :vm, :mc, :models)
                    """,
                    {
                        "sid": session_id,
                        "prompt": prompt,
                        "smode": mode,
                        "vm": voting_mode,
                        "mc": len(models),
                        "models": json.dumps(models),
                    },
                )
                await conn.commit()
        except Exception as e:
            logger.warning("Failed to log session %s: %s", session_id, e)

    async def log_model_response(
        self,
        session_id: str,
        model_name: str,
        temperature: float,
        response_text: str,
        latency_ms: int,
        success: bool = True,
        error_msg: str = "",
    ) -> None:
        """Log an individual model response."""
        if not self._pool.is_available:
            return
        try:
            async with self._pool.pool.acquire() as conn:
                cursor = conn.cursor()
                await cursor.execute(
                    """
                    INSERT INTO CONGRESS_MODEL_RESPONSES
                        (id, session_id, model_name, temperature, response_text,
                         latency_ms, success, error_msg)
                    VALUES (:id, :sid, :model, :temp, :resp, :lat, :suc, :err)
                    """,
                    {
                        "id": str(uuid.uuid4()),
                        "sid": session_id,
                        "model": model_name,
                        "temp": temperature,
                        "resp": response_text,
                        "lat": latency_ms,
                        "suc": 1 if success else 0,
                        "err": error_msg or None,
                    },
                )
                await conn.commit()
        except Exception as e:
            logger.warning("Failed to log model response: %s", e)

    async def log_vote(
        self,
        session_id: str,
        voting_mode: str,
        winner_model: str,
        consensus: float,
        cluster_count: int = 0,
        vote_data: Dict | None = None,
    ) -> None:
        """Log a voting outcome."""
        if not self._pool.is_available:
            return
        try:
            async with self._pool.pool.acquire() as conn:
                cursor = conn.cursor()
                await cursor.execute(
                    """
                    INSERT INTO CONGRESS_VOTES
                        (id, session_id, voting_mode, winner_model, consensus,
                         cluster_count, vote_data)
                    VALUES (:id, :sid, :vm, :wm, :con, :cc, :vd)
                    """,
                    {
                        "id": str(uuid.uuid4()),
                        "sid": session_id,
                        "vm": voting_mode,
                        "wm": winner_model,
                        "con": consensus,
                        "cc": cluster_count,
                        "vd": json.dumps(vote_data) if vote_data else None,
                    },
                )
                await conn.commit()
        except Exception as e:
            logger.warning("Failed to log vote: %s", e)

    async def log_debate_round(
        self,
        session_id: str,
        round_num: int,
        model_name: str,
        response: str,
        cluster_id: int = 0,
        indecisive: bool = False,
        conviction_score: float = 1.0,
    ) -> None:
        """Log a single debate round entry."""
        if not self._pool.is_available:
            return
        try:
            async with self._pool.pool.acquire() as conn:
                cursor = conn.cursor()
                await cursor.execute(
                    """
                    INSERT INTO CONGRESS_DEBATES
                        (id, session_id, round_num, model_name, response,
                         cluster_id, indecisive, conviction_score)
                    VALUES (:id, :sid, :rn, :mn, :resp, :cid, :ind, :cs)
                    """,
                    {
                        "id": str(uuid.uuid4()),
                        "sid": session_id,
                        "rn": round_num,
                        "mn": model_name,
                        "resp": response,
                        "cid": cluster_id,
                        "ind": 1 if indecisive else 0,
                        "cs": conviction_score,
                    },
                )
                await conn.commit()
        except Exception as e:
            logger.warning("Failed to log debate round: %s", e)

    async def log_precedent_cited(
        self,
        session_id: str,
        precedent_id: str,
        action: str,
        similarity: float,
        disposition: str = "",
    ) -> None:
        """Log a precedent citation event."""
        if not self._pool.is_available:
            return
        try:
            async with self._pool.pool.acquire() as conn:
                cursor = conn.cursor()
                await cursor.execute(
                    """
                    INSERT INTO CONGRESS_EVENTS
                        (id, session_id, event_type, sequence_num, event_data)
                    VALUES (:id, :sid, :et, :seq, :ed)
                    """,
                    {
                        "id": str(uuid.uuid4()),
                        "sid": session_id,
                        "et": "PRECEDENT_CITED",
                        "seq": self._next_seq(session_id),
                        "ed": json.dumps({
                            "precedent_id": precedent_id,
                            "action": action,
                            "similarity": similarity,
                            "disposition": disposition,
                        }),
                    },
                )
                await conn.commit()
        except Exception as e:
            logger.warning("Failed to log precedent citation: %s", e)

    async def _flush_loop(self) -> None:
        """Background loop that periodically flushes queued events."""
        while self._running:
            try:
                await asyncio.sleep(self._flush_interval)
                await self._flush_batch()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Flush loop error: %s", e)

    async def _flush_batch(self) -> None:
        """Drain the queue and write events to Oracle in a batch."""
        if not self._pool.is_available or self._queue.empty():
            return

        batch: List[Event] = []
        while not self._queue.empty() and len(batch) < self._batch_size:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        if not batch:
            return

        try:
            async with self._pool.pool.acquire() as conn:
                cursor = conn.cursor()
                for event in batch:
                    seq = self._next_seq(event.session_id)
                    await cursor.execute(
                        """
                        INSERT INTO CONGRESS_EVENTS
                            (id, session_id, event_type, sequence_num, event_data)
                        VALUES (:id, :sid, :et, :seq, :ed)
                        """,
                        {
                            "id": event.id,
                            "sid": event.session_id or None,
                            "et": event.event_type,
                            "seq": seq,
                            "ed": json.dumps(event.event_data),
                        },
                    )
                await conn.commit()
                logger.debug("Flushed %d events to Oracle", len(batch))
        except Exception as e:
            logger.warning("Failed to flush %d events: %s", len(batch), e)

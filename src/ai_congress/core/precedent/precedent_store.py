"""
PrecedentStore — stores and retrieves consensus rulings from Oracle data lake.

Uses CONGRESS_PRECEDENTS table with VECTOR(384, FLOAT32) column for
similarity search via Oracle AI Vector Search.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np

from ...datalake.connection import OraclePoolManager

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384


@dataclass
class Precedent:
    """A stored precedent ruling."""
    id: str
    session_id: str
    query_text: str
    ruling_text: str
    domain: str
    consensus: float
    models_used: List[str]
    debate_rounds: int = 0
    similarity: float = 0.0
    superseded_by: Optional[str] = None
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "query_text": self.query_text,
            "ruling_text": self.ruling_text,
            "domain": self.domain,
            "consensus": self.consensus,
            "models_used": self.models_used,
            "debate_rounds": self.debate_rounds,
            "similarity": self.similarity,
            "superseded_by": self.superseded_by,
            "created_at": self.created_at,
        }


class PrecedentStore:
    """Stores and retrieves precedent rulings from Oracle data lake."""

    def __init__(self, pool_manager: OraclePoolManager, embedding_generator):
        self._pool = pool_manager
        self._embedder = embedding_generator

    async def store_precedent(
        self,
        session_id: str,
        query_text: str,
        ruling_text: str,
        domain: str,
        consensus: float,
        models_used: List[str],
        vote_data: Dict[str, Any],
        debate_rounds: int = 0,
    ) -> str:
        """Embed query and insert precedent. Returns precedent ID."""
        if not self._pool.is_available:
            logger.warning("Oracle unavailable, skipping precedent storage")
            return ""

        precedent_id = str(uuid.uuid4())

        try:
            embedding = await asyncio.to_thread(self._embedder.generate_embedding, query_text)
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

            async with self._pool.pool.acquire() as conn:
                cursor = conn.cursor()
                await cursor.execute(
                    """
                    INSERT INTO CONGRESS_PRECEDENTS
                        (id, session_id, query_text, ruling_text, domain,
                         consensus, models_used, vote_data, debate_rounds, embedding)
                    VALUES (:id, :sid, :qt, :rt, :dom, :con, :mu, :vd, :dr,
                            VECTOR(:emb, 384, FLOAT32))
                    """,
                    {
                        "id": precedent_id,
                        "sid": session_id,
                        "qt": query_text,
                        "rt": ruling_text,
                        "dom": domain,
                        "con": consensus,
                        "mu": json.dumps(models_used),
                        "vd": json.dumps(vote_data) if vote_data else None,
                        "dr": debate_rounds,
                        "emb": embedding_list,
                    },
                )
                await conn.commit()
            logger.info("Stored precedent %s (domain=%s, consensus=%.3f)", precedent_id, domain, consensus)
            return precedent_id

        except Exception as e:
            logger.warning("Failed to store precedent: %s", e)
            return ""

    async def search_precedents(
        self,
        query_text: str,
        domain: Optional[str] = None,
        top_k: int = 3,
        min_similarity: float = 0.75,
    ) -> List[Precedent]:
        """Embed query and vector-search for similar past rulings."""
        if not self._pool.is_available:
            return []

        try:
            embedding = await asyncio.to_thread(self._embedder.generate_embedding, query_text)
            embedding_list = embedding.tolist() if isinstance(embedding, np.ndarray) else embedding

            domain_filter = "AND domain = :dom" if domain else ""

            sql = f"""
                SELECT
                    id, session_id, query_text, ruling_text, domain,
                    consensus, models_used, debate_rounds, superseded_by,
                    created_at,
                    VECTOR_DISTANCE(embedding, VECTOR(:emb, 384, FLOAT32), COSINE) as distance
                FROM CONGRESS_PRECEDENTS
                WHERE superseded_by IS NULL
                {domain_filter}
                ORDER BY distance
                FETCH FIRST :topk ROWS ONLY
            """

            params: dict = {"emb": embedding_list, "topk": top_k}
            if domain:
                params["dom"] = domain

            async with self._pool.pool.acquire() as conn:
                cursor = conn.cursor()
                await cursor.execute(sql, params)
                rows = await cursor.fetchall()

            precedents = []
            for row in rows:
                similarity = 1.0 - float(row[10])
                if similarity < min_similarity:
                    continue
                precedents.append(Precedent(
                    id=row[0],
                    session_id=row[1],
                    query_text=row[2],
                    ruling_text=row[3],
                    domain=row[4] or "general",
                    consensus=float(row[5]),
                    models_used=json.loads(row[6]) if row[6] else [],
                    debate_rounds=int(row[7]) if row[7] else 0,
                    superseded_by=row[8],
                    created_at=row[9].isoformat() if row[9] else "",
                    similarity=similarity,
                ))

            logger.info("Found %d precedents for query (domain=%s)", len(precedents), domain)
            return precedents

        except Exception as e:
            logger.warning("Precedent search failed: %s", e)
            return []

    async def supersede(self, old_id: str, new_id: str) -> None:
        """Mark an old precedent as superseded by a new one."""
        if not self._pool.is_available:
            return
        try:
            async with self._pool.pool.acquire() as conn:
                cursor = conn.cursor()
                await cursor.execute(
                    """
                    UPDATE CONGRESS_PRECEDENTS
                    SET superseded_by = :new_id
                    WHERE id = :old_id AND superseded_by IS NULL
                    """,
                    {"new_id": new_id, "old_id": old_id},
                )
                await conn.commit()
            logger.info("Superseded precedent %s -> %s", old_id, new_id)
        except Exception as e:
            logger.warning("Failed to supersede precedent: %s", e)

    async def list_precedents(
        self, limit: int = 50, offset: int = 0, domain: Optional[str] = None
    ) -> List[Precedent]:
        """List precedents, most recent first."""
        if not self._pool.is_available:
            return []
        try:
            domain_filter = "WHERE domain = :dom" if domain else ""
            sql = f"""
                SELECT id, session_id, query_text, ruling_text, domain,
                       consensus, models_used, debate_rounds, superseded_by, created_at
                FROM CONGRESS_PRECEDENTS
                {domain_filter}
                ORDER BY created_at DESC
                OFFSET :off ROWS FETCH NEXT :lim ROWS ONLY
            """
            params: dict = {"off": offset, "lim": limit}
            if domain:
                params["dom"] = domain

            async with self._pool.pool.acquire() as conn:
                cursor = conn.cursor()
                await cursor.execute(sql, params)
                rows = await cursor.fetchall()

            return [
                Precedent(
                    id=row[0],
                    session_id=row[1],
                    query_text=row[2],
                    ruling_text=row[3],
                    domain=row[4] or "general",
                    consensus=float(row[5]),
                    models_used=json.loads(row[6]) if row[6] else [],
                    debate_rounds=int(row[7]) if row[7] else 0,
                    superseded_by=row[8],
                    created_at=row[9].isoformat() if row[9] else "",
                )
                for row in rows
            ]
        except Exception as e:
            logger.warning("Failed to list precedents: %s", e)
            return []

    async def get_precedent(self, precedent_id: str) -> Optional[Precedent]:
        """Get a single precedent by ID."""
        if not self._pool.is_available:
            return None
        try:
            async with self._pool.pool.acquire() as conn:
                cursor = conn.cursor()
                await cursor.execute(
                    """
                    SELECT id, session_id, query_text, ruling_text, domain,
                           consensus, models_used, debate_rounds, superseded_by, created_at
                    FROM CONGRESS_PRECEDENTS WHERE id = :pid
                    """,
                    {"pid": precedent_id},
                )
                row = await cursor.fetchone()
            if not row:
                return None
            return Precedent(
                id=row[0],
                session_id=row[1],
                query_text=row[2],
                ruling_text=row[3],
                domain=row[4] or "general",
                consensus=float(row[5]),
                models_used=json.loads(row[6]) if row[6] else [],
                debate_rounds=int(row[7]) if row[7] else 0,
                superseded_by=row[8],
                created_at=row[9].isoformat() if row[9] else "",
            )
        except Exception as e:
            logger.warning("Failed to get precedent %s: %s", precedent_id, e)
            return None

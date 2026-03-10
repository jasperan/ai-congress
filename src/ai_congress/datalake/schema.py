"""
DDL for CONGRESS_* data lake tables in Oracle 26ai Free.
"""

import logging

from .connection import OraclePoolManager

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "1.0.0"

DDL_STATEMENTS = [
    # Meta table for schema versioning
    """
    CREATE TABLE CONGRESS_META (
        meta_key    VARCHAR2(100) PRIMARY KEY,
        meta_value  VARCHAR2(500),
        updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    # Sessions — one per chat request
    """
    CREATE TABLE CONGRESS_SESSIONS (
        session_id   VARCHAR2(36) PRIMARY KEY,
        prompt       CLOB,
        swarm_mode   VARCHAR2(50),
        voting_mode  VARCHAR2(20),
        model_count  NUMBER(5),
        models_used  CLOB,
        created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    # Append-only event log
    """
    CREATE TABLE CONGRESS_EVENTS (
        id            VARCHAR2(36) PRIMARY KEY,
        session_id    VARCHAR2(36),
        event_type    VARCHAR2(50) NOT NULL,
        sequence_num  NUMBER(10),
        event_data    CLOB,
        created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT chk_events_json CHECK (event_data IS JSON)
    )
    """,
    # Individual model responses
    """
    CREATE TABLE CONGRESS_MODEL_RESPONSES (
        id              VARCHAR2(36) PRIMARY KEY,
        session_id      VARCHAR2(36) NOT NULL,
        model_name      VARCHAR2(200) NOT NULL,
        temperature     NUMBER(4,2),
        response_text   CLOB,
        latency_ms      NUMBER(12),
        success         NUMBER(1) DEFAULT 1,
        error_msg       VARCHAR2(2000),
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    # Voting outcomes
    """
    CREATE TABLE CONGRESS_VOTES (
        id              VARCHAR2(36) PRIMARY KEY,
        session_id      VARCHAR2(36) NOT NULL,
        voting_mode     VARCHAR2(20),
        winner_model    VARCHAR2(200),
        consensus       NUMBER(5,3),
        cluster_count   NUMBER(5),
        vote_data       CLOB,
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT chk_votes_json CHECK (vote_data IS JSON)
    )
    """,
    # Debate rounds
    """
    CREATE TABLE CONGRESS_DEBATES (
        id               VARCHAR2(36) PRIMARY KEY,
        session_id       VARCHAR2(36) NOT NULL,
        round_num        NUMBER(3) NOT NULL,
        model_name       VARCHAR2(200) NOT NULL,
        response         CLOB,
        cluster_id       NUMBER(5),
        indecisive       NUMBER(1) DEFAULT 0,
        conviction_score NUMBER(5,3),
        created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    # Precedent rulings (stare decisis)
    """
    CREATE TABLE CONGRESS_PRECEDENTS (
        id              VARCHAR2(36) PRIMARY KEY,
        session_id      VARCHAR2(36) NOT NULL,
        query_text      CLOB NOT NULL,
        ruling_text     CLOB NOT NULL,
        domain          VARCHAR2(50),
        consensus       NUMBER(5,3) NOT NULL,
        models_used     CLOB,
        vote_data       CLOB,
        debate_rounds   NUMBER(3) DEFAULT 0,
        superseded_by   VARCHAR2(36),
        embedding       VECTOR(384, FLOAT32),
        created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
]

INDEX_STATEMENTS = [
    "CREATE INDEX idx_events_session ON CONGRESS_EVENTS(session_id)",
    "CREATE INDEX idx_events_type ON CONGRESS_EVENTS(event_type)",
    "CREATE INDEX idx_events_created ON CONGRESS_EVENTS(created_at)",
    "CREATE INDEX idx_model_resp_session ON CONGRESS_MODEL_RESPONSES(session_id)",
    "CREATE INDEX idx_model_resp_model ON CONGRESS_MODEL_RESPONSES(model_name)",
    "CREATE INDEX idx_votes_session ON CONGRESS_VOTES(session_id)",
    "CREATE INDEX idx_debates_session ON CONGRESS_DEBATES(session_id)",
    "CREATE INDEX idx_sessions_created ON CONGRESS_SESSIONS(created_at)",
    "CREATE INDEX idx_precedents_domain ON CONGRESS_PRECEDENTS(domain)",
    "CREATE INDEX idx_precedents_superseded ON CONGRESS_PRECEDENTS(superseded_by)",
    "CREATE INDEX idx_precedents_session ON CONGRESS_PRECEDENTS(session_id)",
]

VECTOR_INDEX_STATEMENTS = [
    """
    CREATE VECTOR INDEX idx_precedents_vec
        ON CONGRESS_PRECEDENTS(embedding)
        ORGANIZATION NEIGHBOR PARTITIONS
        WITH DISTANCE COSINE
    """,
]


async def init_schema(pool_manager: OraclePoolManager) -> bool:
    """Create all CONGRESS_* tables if they don't exist. Returns True on success."""
    if not pool_manager.is_available:
        logger.warning("Oracle pool not available, skipping schema init")
        return False

    try:
        async with pool_manager.pool.acquire() as conn:
            cursor = conn.cursor()

            # Check which tables already exist
            await cursor.execute(
                "SELECT table_name FROM user_tables WHERE table_name LIKE 'CONGRESS_%'"
            )
            existing = {row[0] for row in await cursor.fetchall()}
            logger.info(f"Existing CONGRESS_* tables: {existing or 'none'}")

            # Create missing tables
            for ddl in DDL_STATEMENTS:
                table_name = ddl.split("CREATE TABLE")[1].split("(")[0].strip()
                if table_name in existing:
                    continue
                try:
                    await cursor.execute(ddl)
                    logger.info(f"Created table {table_name}")
                except Exception as e:
                    if "ORA-00955" in str(e):  # already exists
                        continue
                    logger.error(f"Failed to create {table_name}: {e}")

            # Create indexes (ignore if already exist)
            for idx_sql in INDEX_STATEMENTS:
                try:
                    await cursor.execute(idx_sql)
                except Exception as e:
                    if "ORA-00955" in str(e) or "ORA-01408" in str(e):
                        continue
                    logger.warning(f"Index creation warning: {e}")

            # Create vector indexes (ignore if already exist)
            for vidx_sql in VECTOR_INDEX_STATEMENTS:
                try:
                    await cursor.execute(vidx_sql)
                except Exception as e:
                    if "ORA-00955" in str(e) or "ORA-01408" in str(e):
                        continue
                    logger.warning(f"Vector index creation warning: {e}")

            # Upsert schema version
            await cursor.execute(
                """
                MERGE INTO CONGRESS_META t
                USING (SELECT 'schema_version' AS meta_key FROM DUAL) s
                ON (t.meta_key = s.meta_key)
                WHEN MATCHED THEN UPDATE SET
                    meta_value = :ver, updated_at = CURRENT_TIMESTAMP
                WHEN NOT MATCHED THEN INSERT
                    (meta_key, meta_value) VALUES ('schema_version', :ver)
                """,
                {"ver": SCHEMA_VERSION},
            )
            await conn.commit()
            logger.info(f"Schema initialized (version {SCHEMA_VERSION})")
            return True

    except Exception as e:
        logger.error(f"Schema initialization failed: {e}")
        return False

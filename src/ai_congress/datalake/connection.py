"""
Async Oracle connection pool manager for FreePDB1 (thin mode).
"""

import os
import logging

import oracledb

logger = logging.getLogger(__name__)

oracledb.defaults.fetch_lobs = False


class OraclePoolManager:
    """Manages an async connection pool to Oracle 26ai Free."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        service: str | None = None,
        user: str | None = None,
        password: str | None = None,
        pool_min: int | None = None,
        pool_max: int | None = None,
    ):
        self.host = host or os.getenv("ORACLE_HOST", "localhost")
        self.port = port or int(os.getenv("ORACLE_PORT", "1521"))
        self.service = service or os.getenv("ORACLE_SERVICE", "FREEPDB1")
        self.user = user or os.getenv("ORACLE_USER", "ADMIN")
        self.password = password or os.getenv("ORACLE_PASSWORD", "")
        self.pool_min = pool_min or int(os.getenv("ORACLE_POOL_MIN", "2"))
        self.pool_max = pool_max or int(os.getenv("ORACLE_POOL_MAX", "5"))
        self._pool: oracledb.AsyncConnectionPool | None = None

    @property
    def dsn(self) -> str:
        return f"{self.host}:{self.port}/{self.service}"

    async def start(self) -> None:
        """Create the async connection pool."""
        if self._pool is not None:
            return
        try:
            self._pool = oracledb.create_pool_async(
                user=self.user,
                password=self.password,
                dsn=self.dsn,
                min=self.pool_min,
                max=self.pool_max,
            )
            logger.info(f"Oracle pool created: {self.dsn} (min={self.pool_min}, max={self.pool_max})")
        except Exception as e:
            logger.error(f"Failed to create Oracle pool: {e}")
            self._pool = None
            raise

    async def stop(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            try:
                await self._pool.close(force=True)
                logger.info("Oracle pool closed")
            except Exception as e:
                logger.error(f"Error closing Oracle pool: {e}")
            finally:
                self._pool = None

    @property
    def pool(self) -> oracledb.AsyncConnectionPool | None:
        return self._pool

    @property
    def is_available(self) -> bool:
        return self._pool is not None

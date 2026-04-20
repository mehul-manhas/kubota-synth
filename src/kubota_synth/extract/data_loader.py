"""Data loader that pulls real rows from MSSQL for training SDV models."""

from __future__ import annotations

import logging
import time

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from kubota_synth.config import ConnectionConfig

logger = logging.getLogger(__name__)


class DataLoader:
    """Small helper around SQLAlchemy for loading sampled data from MSSQL."""

    def __init__(self, conn: ConnectionConfig, schema: str | None = None) -> None:
        self.conn = conn
        self.schema = schema or conn.schema or "dbo"
        self._engine: Engine | None = None

    # -- engine lifecycle ---------------------------------------------------

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            logger.debug(
                "DataLoader: creating engine for %s/%s schema=%s",
                self.conn.server,
                self.conn.database,
                self.schema,
            )
            self._engine = create_engine(self.conn.sqlalchemy_url(), pool_pre_ping=True)
        return self._engine

    def close(self) -> None:
        if self._engine is not None:
            logger.debug("DataLoader: disposing engine for %s/%s", self.conn.server, self.conn.database)
            try:
                self._engine.dispose()
            finally:
                self._engine = None

    def __enter__(self) -> DataLoader:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- queries ------------------------------------------------------------

    def row_count(self, table: str) -> int:
        sql = text(f"SELECT COUNT(*) FROM [{self.schema}].[{table}]")
        with self.engine.connect() as cn:
            value = cn.execute(sql).scalar()
        return int(value or 0)

    def load(self, table: str, sample_size: int | None = None) -> pd.DataFrame:
        """Load (sampled) rows from ``schema.table``.

        If ``sample_size`` is ``None`` or the total row count is smaller than
        ``sample_size``, the full table is loaded. Otherwise MSSQL's
        ``TABLESAMPLE`` clause is used for efficient block-level sampling:

            SELECT TOP (<sample>) * FROM [schema].[table] TABLESAMPLE (<pct> PERCENT)
        """
        t0 = time.perf_counter()
        total = self.row_count(table)
        logger.info("Table %s.%s has %d rows.", self.schema, table, total)

        if sample_size is None or total <= sample_size:
            sql = f"SELECT * FROM [{self.schema}].[{table}]"
            logger.info("Loading entire table %s (%d rows).", table, total)
            logger.debug("DataLoader SQL: %s", sql[:500] + ("..." if len(sql) > 500 else ""))
            df = pd.read_sql(sql, self.engine)
            logger.info(
                "Loaded %d rows from %s in %.2fs (full table).",
                len(df),
                table,
                time.perf_counter() - t0,
            )
            return df

        # Overshoot by 20 % to compensate for TABLESAMPLE variance, then cap.
        pct = min(100.0, (sample_size / max(total, 1)) * 100 * 1.2)
        sql = (
            f"SELECT TOP ({int(sample_size)}) * "
            f"FROM [{self.schema}].[{table}] TABLESAMPLE ({pct:.4f} PERCENT)"
        )
        logger.info(
            "Sampling %s: target=%d rows (%.4f%% TABLESAMPLE).", table, sample_size, pct
        )
        logger.debug("DataLoader SQL: %s", sql)
        df = pd.read_sql(sql, self.engine)

        if len(df) < sample_size * 0.5:
            logger.warning(
                "TABLESAMPLE returned fewer rows than expected (%d of target %d) -- "
                "falling back to TOP+ORDER BY NEWID() sampling.",
                len(df),
                sample_size,
            )
            sql = (
                f"SELECT TOP ({int(sample_size)}) * FROM [{self.schema}].[{table}] "
                "ORDER BY NEWID()"
            )
            logger.debug("DataLoader fallback SQL: %s", sql)
            df = pd.read_sql(sql, self.engine)

        logger.info(
            "Loaded %d rows from %s in %.2fs.",
            len(df),
            table,
            time.perf_counter() - t0,
        )
        return df

"""Append synthetic rows to the target MSSQL database."""

from __future__ import annotations

import logging
import math
import time

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from kubota_synth.config import ConnectionConfig, ProjectConfig

logger = logging.getLogger(__name__)


class MSSQLWriter:
    """Bulk-insert helper tuned for MSSQL via pyodbc + fast_executemany."""

    def __init__(self, conn: ConnectionConfig, schema: str | None = None) -> None:
        self.conn = conn
        self.schema = schema or conn.schema or "dbo"
        self._engine: Engine | None = None

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            # fast_executemany=True is THE optimization for bulk inserts on MSSQL
            # via pyodbc -- the difference between ~100 rows/sec and 50k rows/sec.
            logger.debug(
                "MSSQLWriter: creating engine for %s/%s (fast_executemany)",
                self.conn.server,
                self.conn.database,
            )
            self._engine = create_engine(
                self.conn.sqlalchemy_url(),
                fast_executemany=True,
                pool_pre_ping=True,
            )
        return self._engine

    def close(self) -> None:
        if self._engine is not None:
            logger.debug("MSSQLWriter: disposing engine for %s/%s", self.conn.server, self.conn.database)
            try:
                self._engine.dispose()
            finally:
                self._engine = None

    def __enter__(self) -> MSSQLWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def table_exists(self, table: str) -> bool:
        sql = text(
            """
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
            WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
            """
        )
        with self.engine.connect() as cn:
            count = cn.execute(sql, {"schema": self.schema, "table": table}).scalar()
        return int(count or 0) > 0

    def append(self, table: str, df: pd.DataFrame, batch_size: int = 10_000) -> int:
        """Append ``df`` to ``schema.table``. Returns rows written."""
        if df is None or df.empty:
            logger.warning("append() called with empty DataFrame for %s -- skipping.", table)
            return 0

        exists = self.table_exists(table)
        logger.debug(
            "append: target [%s].[%s] exists=%s rows=%d batch_size=%d",
            self.schema,
            table,
            exists,
            len(df),
            batch_size,
        )
        if not exists:
            raise RuntimeError(
                f"Target table [{self.schema}].[{table}] does not exist. "
                "kubota-synth APPENDS to existing tables -- it will never create them. "
                "Create it first with a shape-copy from the source, e.g.:\n\n"
                f"  SELECT TOP 0 *\n"
                f"  INTO {self.schema}.{table}\n"
                f"  FROM [<source-server>].[<source-db>].{self.schema}.{table};\n"
            )

        n = len(df)
        batches = max(1, math.ceil(n / int(batch_size)))
        t0 = time.perf_counter()
        logger.info(
            "Appending %d row(s) to [%s].[%s] in ~%d batch(es) of up to %d.",
            n,
            self.schema,
            table,
            batches,
            int(batch_size),
        )
        try:
            df.to_sql(
                name=table,
                con=self.engine,
                schema=self.schema,
                if_exists="append",
                index=False,
                chunksize=int(batch_size),
                method=None,
            )
        except Exception:
            logger.exception("Failed to append %d rows to %s.%s.", len(df), self.schema, table)
            raise

        elapsed = time.perf_counter() - t0
        rps = (n / elapsed) if elapsed > 0 else float(n)
        logger.info(
            "Appended %d rows to [%s].[%s] in %.2fs (~%.0f rows/s).",
            n,
            self.schema,
            table,
            elapsed,
            rps,
        )
        return len(df)


def write_synthetic(table_name: str, df: pd.DataFrame, cfg: ProjectConfig) -> int:
    """High-level helper used by the CLI to write synthetic rows to MSSQL."""
    logger.debug(
        "write_synthetic: table=%s target_schema=%s batch_size=%s",
        table_name,
        cfg.target_schema,
        cfg.batch_size,
    )
    writer = MSSQLWriter(cfg.target, schema=cfg.target_schema)
    try:
        return writer.append(table_name, df, batch_size=cfg.batch_size)
    finally:
        writer.close()

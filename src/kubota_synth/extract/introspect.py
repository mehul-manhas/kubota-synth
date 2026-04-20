"""MSSQL schema introspection and SDV metadata builders.

This module converts the live MSSQL schema into SDV V1 metadata dicts. The
output can be fed directly into ``SingleTableMetadata.load_from_dict`` or the
multi-table ``Metadata.load_from_dict``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from kubota_synth.config import ConnectionConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MSSQL -> SDV sdtype mapping
# ---------------------------------------------------------------------------

MSSQL_TO_SDTYPE: dict[str, str] = {
    # numerical
    "int": "numerical",
    "bigint": "numerical",
    "smallint": "numerical",
    "tinyint": "numerical",
    "decimal": "numerical",
    "numeric": "numerical",
    "float": "numerical",
    "real": "numerical",
    "money": "numerical",
    "smallmoney": "numerical",
    # datetime
    "datetime": "datetime",
    "datetime2": "datetime",
    "smalldatetime": "datetime",
    "date": "datetime",
    "time": "datetime",
    "datetimeoffset": "datetime",
    # strings (may be flipped to "text" if high cardinality)
    "varchar": "categorical",
    "nvarchar": "categorical",
    "char": "categorical",
    "nchar": "categorical",
    # long strings
    "text": "text",
    "ntext": "text",
    # boolean
    "bit": "boolean",
    # id
    "uniqueidentifier": "id",
}

CATEGORICAL_CARDINALITY_LIMIT = 200

_STRING_DATATYPES = {"varchar", "nvarchar", "char", "nchar"}


# ---------------------------------------------------------------------------
# Introspector
# ---------------------------------------------------------------------------


class MSSQLIntrospector:
    """Query MSSQL system views to build SDV metadata."""

    def __init__(self, conn: ConnectionConfig, schema: str | None = None) -> None:
        self.conn = conn
        self.schema = schema or conn.schema or "dbo"
        self._engine: Engine | None = None

    # -- engine lifecycle ---------------------------------------------------

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            self._engine = create_engine(self.conn.sqlalchemy_url(), pool_pre_ping=True)
        return self._engine

    def close(self) -> None:
        if self._engine is not None:
            try:
                self._engine.dispose()
            finally:
                self._engine = None

    def __enter__(self) -> MSSQLIntrospector:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- schema queries -----------------------------------------------------

    def list_columns(self, table: str) -> list[dict[str, Any]]:
        """Return metadata for every column in ``schema.table``."""
        sql = text(
            """
            SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, CHARACTER_MAXIMUM_LENGTH
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = :schema AND TABLE_NAME = :table
            ORDER BY ORDINAL_POSITION
            """
        )
        with self.engine.connect() as cn:
            rows = cn.execute(sql, {"schema": self.schema, "table": table}).fetchall()

        return [
            {
                "column_name": r[0],
                "data_type": (r[1] or "").lower(),
                "is_nullable": (r[2] or "YES").upper() == "YES",
                "char_max_length": r[3],
            }
            for r in rows
        ]

    def get_primary_key(self, table: str) -> str | None:
        """Return the primary key column name (first if composite)."""
        sql = text(
            """
            SELECT kcu.COLUMN_NAME, kcu.ORDINAL_POSITION
            FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
            JOIN INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
              ON tc.CONSTRAINT_NAME = kcu.CONSTRAINT_NAME
             AND tc.TABLE_SCHEMA   = kcu.TABLE_SCHEMA
             AND tc.TABLE_NAME     = kcu.TABLE_NAME
            WHERE tc.CONSTRAINT_TYPE = 'PRIMARY KEY'
              AND tc.TABLE_SCHEMA   = :schema
              AND tc.TABLE_NAME     = :table
            ORDER BY kcu.ORDINAL_POSITION
            """
        )
        with self.engine.connect() as cn:
            rows = cn.execute(sql, {"schema": self.schema, "table": table}).fetchall()

        if not rows:
            return None
        if len(rows) > 1:
            cols = ", ".join(r[0] for r in rows)
            logger.warning(
                "Table %s.%s has a composite primary key (%s). SDV OSS does not "
                "support composite PKs -- using the first column (%s).",
                self.schema,
                table,
                cols,
                rows[0][0],
            )
        return rows[0][0]

    def get_foreign_keys(self, tables: list[str]) -> list[dict[str, str]]:
        """Return FK relationships between the given tables (ignores others)."""
        if not tables:
            return []

        sql = text(
            """
            SELECT
                ps.name   AS parent_schema,
                pt.name   AS parent_table,
                pc.name   AS parent_column,
                cs.name   AS child_schema,
                ct.name   AS child_table,
                cc.name   AS child_column
            FROM sys.foreign_keys fk
            JOIN sys.foreign_key_columns fkc ON fk.object_id = fkc.constraint_object_id
            JOIN sys.tables    pt ON fkc.referenced_object_id = pt.object_id
            JOIN sys.schemas   ps ON pt.schema_id = ps.schema_id
            JOIN sys.columns   pc ON fkc.referenced_object_id = pc.object_id
                                 AND fkc.referenced_column_id = pc.column_id
            JOIN sys.tables    ct ON fkc.parent_object_id = ct.object_id
            JOIN sys.schemas   cs ON ct.schema_id = cs.schema_id
            JOIN sys.columns   cc ON fkc.parent_object_id = cc.object_id
                                 AND fkc.parent_column_id = cc.column_id
            WHERE ps.name = :schema
              AND cs.name = :schema
            """
        )
        wanted = set(tables)
        with self.engine.connect() as cn:
            rows = cn.execute(sql, {"schema": self.schema}).fetchall()

        relationships: list[dict[str, str]] = []
        for parent_schema, parent_table, parent_col, _child_schema, child_table, child_col in rows:
            if parent_table in wanted and child_table in wanted:
                relationships.append(
                    {
                        "parent_table_name": parent_table,
                        "parent_primary_key": parent_col,
                        "child_table_name": child_table,
                        "child_foreign_key": child_col,
                    }
                )
        return relationships

    def estimate_cardinality(self, table: str, column: str) -> int:
        """Return ``COUNT(DISTINCT [column])`` for ``schema.table``."""
        sql = text(
            f"SELECT COUNT(DISTINCT [{column}]) FROM [{self.schema}].[{table}]"
        )
        with self.engine.connect() as cn:
            value = cn.execute(sql).scalar()
        return int(value or 0)

    # -- metadata builder ---------------------------------------------------

    def build_table_metadata(
        self,
        table: str,
        pk: str | None = None,
        override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build an SDV V1 single-table metadata dict for ``schema.table``."""
        override = override or {}
        column_overrides = (override.get("columns") or {}) if isinstance(override, dict) else {}

        columns_info = self.list_columns(table)
        if pk is None:
            pk = self.get_primary_key(table)

        columns: dict[str, dict[str, Any]] = {}
        for col in columns_info:
            name = col["column_name"]
            data_type = col["data_type"]

            sdtype = MSSQL_TO_SDTYPE.get(data_type, "categorical")

            # String columns: flip to "text" if cardinality blows past the limit.
            if data_type in _STRING_DATATYPES and sdtype == "categorical":
                try:
                    card = self.estimate_cardinality(table, name)
                    if card > CATEGORICAL_CARDINALITY_LIMIT:
                        logger.debug(
                            "Column %s.%s has cardinality %d -- switching to text sdtype.",
                            table,
                            name,
                            card,
                        )
                        sdtype = "text"
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Could not estimate cardinality for %s.%s: %s", table, name, exc
                    )

            entry: dict[str, Any] = {"sdtype": sdtype}
            if sdtype == "datetime":
                # Provide a reasonable default; users can override in YAML.
                entry["datetime_format"] = "%Y-%m-%d %H:%M:%S"

            columns[name] = entry

        if pk and pk in columns:
            columns[pk]["sdtype"] = "id"

        # Apply user column overrides last so they always win.
        for col_name, col_override in column_overrides.items():
            if not isinstance(col_override, dict):
                continue
            columns.setdefault(col_name, {}).update(col_override)

        metadata: dict[str, Any] = {"columns": columns}
        if pk:
            metadata["primary_key"] = pk
        return metadata


# ---------------------------------------------------------------------------
# Orchestration helper
# ---------------------------------------------------------------------------


def _load_override(overrides_dir: Path, table: str) -> dict[str, Any]:
    path = overrides_dir / f"{table}.yaml"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        if not isinstance(data, dict):
            logger.warning("Override file %s is not a mapping -- ignoring.", path)
            return {}
        return data
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load override file %s: %s", path, exc)
        return {}


def build_sdv_metadata(
    tables: list[str],
    conn: ConnectionConfig,
    schema: str | None = None,
    overrides_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Return a full SDV V1 multi-table metadata dict for ``tables``."""
    overrides_path = Path(overrides_dir) if overrides_dir else None

    introspector = MSSQLIntrospector(conn, schema=schema)
    try:
        tables_meta: dict[str, Any] = {}
        for table in tables:
            override = _load_override(overrides_path, table) if overrides_path else {}
            pk = introspector.get_primary_key(table)
            if pk is None:
                logger.warning(
                    "Table %s.%s has no primary key -- SDV will auto-generate one.",
                    introspector.schema,
                    table,
                )
            table_meta = introspector.build_table_metadata(table, pk=pk, override=override)

            # Merge user-provided constraints (if any) into the table metadata.
            constraints = override.get("constraints") if isinstance(override, dict) else None
            if constraints:
                table_meta["_constraints"] = constraints

            tables_meta[table] = table_meta

        relationships = introspector.get_foreign_keys(tables)
    finally:
        introspector.close()

    return {
        "METADATA_SPEC_VERSION": "V1",
        "tables": tables_meta,
        "relationships": relationships,
    }

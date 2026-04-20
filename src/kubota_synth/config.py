"""Configuration dataclasses and loaders for kubota-synth.

This module is the single source of truth for

* MSSQL connection details (sourced from environment variables),
* per-table synthesizer configuration (sourced from YAML),
* and project-wide defaults (models directory, log level, ...).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import yaml
from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MSSQL connection configuration
# ---------------------------------------------------------------------------


@dataclass
class ConnectionConfig:
    """Connection details for a single MSSQL server/database."""

    server: str
    database: str
    username: str
    password: str
    driver: str = "ODBC Driver 18 for SQL Server"
    schema: str = "dbo"

    @classmethod
    def from_env(cls, prefix: str, schema: str | None = None) -> ConnectionConfig:
        """Build a ``ConnectionConfig`` from ``MSSQL_{prefix}_*`` env vars.

        Parameters
        ----------
        prefix:
            Either ``"SOURCE"`` or ``"TARGET"``.
        schema:
            Optional schema override. If ``None`` the function falls back to
            ``SOURCE_SCHEMA`` / ``TARGET_SCHEMA`` env vars and finally ``"dbo"``.
        """
        load_dotenv(override=False)

        def _require(name: str) -> str:
            value = os.getenv(name)
            if not value:
                raise RuntimeError(
                    f"Environment variable {name!r} is required but not set. "
                    "Did you copy .env.example to .env and fill it in?"
                )
            return value

        server = _require(f"MSSQL_{prefix}_SERVER")
        database = _require(f"MSSQL_{prefix}_DATABASE")
        username = _require(f"MSSQL_{prefix}_USERNAME")
        password = _require(f"MSSQL_{prefix}_PASSWORD")
        driver = os.getenv(f"MSSQL_{prefix}_DRIVER", "ODBC Driver 18 for SQL Server")

        if schema is None:
            schema = os.getenv(f"{prefix}_SCHEMA", "dbo")

        return cls(
            server=server,
            database=database,
            username=username,
            password=password,
            driver=driver,
            schema=schema,
        )

    def odbc_connection_string(self) -> str:
        """Return a raw ODBC connection string (unsafe for URL use as-is)."""
        return (
            f"DRIVER={{{self.driver}}};"
            f"SERVER={self.server};"
            f"DATABASE={self.database};"
            f"UID={self.username};"
            f"PWD={self.password};"
            "Encrypt=yes;"
            "TrustServerCertificate=yes;"
        )

    def sqlalchemy_url(self) -> str:
        """Return a SQLAlchemy URL that correctly escapes the ODBC string."""
        odbc = self.odbc_connection_string()
        return f"mssql+pyodbc:///?odbc_connect={quote_plus(odbc)}"


# ---------------------------------------------------------------------------
# Per-table synthesizer configuration
# ---------------------------------------------------------------------------


@dataclass
class TableConfig:
    """Configuration for a single table to synthesize."""

    name: str
    synthesizer: str = "GaussianCopula"
    sample_rows: int = 10_000
    fit_sample_size: int = 500_000
    random_seed: int = 42

    # Synthesizer-specific kwargs
    epochs: int | None = None
    sequence_key: str | None = None
    sequence_index: str | None = None

    # Catch-all for extra kwargs a user may add in YAML.
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Project-wide configuration
# ---------------------------------------------------------------------------


@dataclass
class ProjectConfig:
    """Top-level project configuration."""

    source: ConnectionConfig
    target: ConnectionConfig
    tables: dict[str, TableConfig]
    relationships: list[dict[str, Any]] = field(default_factory=list)

    # Output settings
    write_mode: str = "append"
    batch_size: int = 10_000
    target_schema: str = "dbo"

    # Paths
    models_dir: Path = field(default_factory=lambda: Path("./artifacts/models"))
    output_dir: Path = field(default_factory=lambda: Path("./artifacts/output"))
    overrides_dir: Path = field(default_factory=lambda: Path("./config/overrides"))

    # Raw defaults dict (kept for downstream tooling that wants to inspect it)
    defaults: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


_KNOWN_TABLE_KEYS = {
    "synthesizer",
    "sample_rows",
    "fit_sample_size",
    "random_seed",
    "epochs",
    "sequence_key",
    "sequence_index",
}


def _merge_table_config(name: str, defaults: dict[str, Any], overrides: dict[str, Any]) -> TableConfig:
    merged: dict[str, Any] = {**defaults, **(overrides or {})}
    known = {k: merged[k] for k in _KNOWN_TABLE_KEYS if k in merged}
    extra = {k: v for k, v in merged.items() if k not in _KNOWN_TABLE_KEYS}
    return TableConfig(name=name, extra=extra, **known)


def load_config(path: str | Path) -> ProjectConfig:
    """Load a synthesizer config YAML and return a populated ``ProjectConfig``.

    Merges ``defaults`` into each table's configuration. Connection details
    are sourced from environment variables (``.env`` file is loaded
    automatically).
    """
    load_dotenv(override=False)

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    defaults = raw.get("defaults", {}) or {}
    tables_raw = raw.get("tables", {}) or {}
    relationships = raw.get("relationships", []) or []
    output = raw.get("output", {}) or {}

    tables: dict[str, TableConfig] = {
        name: _merge_table_config(name, defaults, cfg or {}) for name, cfg in tables_raw.items()
    }

    source_schema = os.getenv("SOURCE_SCHEMA", "dbo")
    target_schema = output.get("target_schema") or os.getenv("TARGET_SCHEMA", "dbo")

    models_dir = Path(os.getenv("MODELS_DIR", "./artifacts/models"))
    output_dir = Path(os.getenv("OUTPUT_DIR", "./artifacts/output"))
    overrides_dir = Path(os.getenv("OVERRIDES_DIR", "./config/overrides"))

    source = ConnectionConfig.from_env("SOURCE", schema=source_schema)
    target = ConnectionConfig.from_env("TARGET", schema=target_schema)

    models_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ProjectConfig(
        source=source,
        target=target,
        tables=tables,
        relationships=list(relationships),
        write_mode=output.get("write_mode", "append"),
        batch_size=int(output.get("batch_size", 10_000)),
        target_schema=target_schema,
        models_dir=models_dir,
        output_dir=output_dir,
        overrides_dir=overrides_dir,
        defaults=dict(defaults),
    )

    logger.debug(
        "Loaded config with %d tables (source=%s target=%s)",
        len(cfg.tables),
        cfg.source.database,
        cfg.target.database,
    )
    return cfg

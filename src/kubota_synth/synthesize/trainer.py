"""Train SDV synthesizers against real MSSQL data and persist them to disk."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from kubota_synth.config import ProjectConfig
from kubota_synth.extract.data_loader import DataLoader
from kubota_synth.synthesize.registry import build_synthesizer

logger = logging.getLogger(__name__)


def _coerce_dtypes(df: pd.DataFrame, table_metadata: dict[str, Any]) -> pd.DataFrame:
    """Coerce pandas dtypes to what SDV expects based on the metadata dict."""
    columns = table_metadata.get("columns", {}) if isinstance(table_metadata, dict) else {}
    for col, spec in columns.items():
        if col not in df.columns or not isinstance(spec, dict):
            continue
        sdtype = spec.get("sdtype")
        if sdtype == "datetime":
            fmt = spec.get("datetime_format")
            try:
                if fmt:
                    df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
                else:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not parse datetimes for column %s: %s", col, exc)
        elif sdtype == "categorical":
            # SDV is happiest with plain object dtype for categoricals.
            if df[col].dtype.name == "category":
                df[col] = df[col].astype(object)
        elif sdtype == "boolean":
            try:
                df[col] = df[col].astype("boolean")
            except Exception:  # noqa: BLE001
                pass
    return df


def train_table(
    table_name: str,
    sdv_metadata: dict[str, Any],
    cfg: ProjectConfig,
) -> Path:
    """Train a synthesizer for ``table_name`` and persist it. Returns model path."""
    if table_name not in cfg.tables:
        raise KeyError(f"Table {table_name!r} is not listed in the config.")

    table_cfg = cfg.tables[table_name]

    tables_meta = sdv_metadata.get("tables", {}) if isinstance(sdv_metadata, dict) else {}
    if table_name not in tables_meta:
        raise KeyError(
            f"Table {table_name!r} is not present in the provided SDV metadata. "
            "Run `kubota-synth introspect` first."
        )
    table_metadata = tables_meta[table_name]

    logger.info("Training %s synthesizer for table '%s'...", table_cfg.synthesizer, table_name)

    with DataLoader(cfg.source, schema=cfg.source.schema) as loader:
        df = loader.load(table_name, sample_size=table_cfg.fit_sample_size)

    if df.empty:
        raise RuntimeError(
            f"No training data loaded for '{table_name}'. Refusing to fit an empty model."
        )

    df = _coerce_dtypes(df, table_metadata)

    synth = build_synthesizer(table_cfg, table_metadata)
    synth.fit(df)

    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    model_path = cfg.models_dir / f"{table_name}.pkl"
    synth.save(str(model_path))

    logger.info(
        "Trained synthesizer for '%s' on %d rows; saved to %s.",
        table_name,
        len(df),
        model_path,
    )
    return model_path

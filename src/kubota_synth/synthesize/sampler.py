"""Sample new synthetic rows from previously trained SDV models."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sdv.sampling import Condition

from kubota_synth.config import ProjectConfig
from kubota_synth.synthesize.registry import SYNTHESIZER_CLASSES

logger = logging.getLogger(__name__)


def _model_path(cfg: ProjectConfig, table_name: str) -> Path:
    path = cfg.models_dir / f"{table_name}.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"No trained model found for '{table_name}' at {path}. "
            "Run `kubota-synth train` first."
        )
    return path


def _load_synth(cfg: ProjectConfig, table_name: str):
    if table_name not in cfg.tables:
        raise KeyError(f"Table {table_name!r} is not listed in the config.")
    name = cfg.tables[table_name].synthesizer
    cls = SYNTHESIZER_CLASSES.get(name)
    if cls is None:
        raise ValueError(f"Unknown synthesizer {name!r} for table {table_name!r}.")
    path = _model_path(cfg, table_name)
    return cls.load(str(path))


def sample_table(
    table_name: str,
    cfg: ProjectConfig,
    rows: int | None = None,
) -> pd.DataFrame:
    """Sample ``rows`` synthetic rows for ``table_name``."""
    table_cfg = cfg.tables[table_name]
    n = int(rows) if rows is not None else int(table_cfg.sample_rows)

    synth = _load_synth(cfg, table_name)

    if table_cfg.synthesizer == "PAR":
        num_sequences = max(1, n // 100)
        logger.info(
            "Sampling %d sequences from PAR synthesizer '%s' (target ~%d rows).",
            num_sequences,
            table_name,
            n,
        )
        df = synth.sample(num_sequences=num_sequences)
    else:
        logger.info("Sampling %d rows from '%s'.", n, table_name)
        df = synth.sample(num_rows=n)

    logger.info("Produced %d synthetic rows for '%s'.", len(df), table_name)
    return df


def sample_conditional(
    table_name: str,
    cfg: ProjectConfig,
    conditions: list[dict],
) -> pd.DataFrame:
    """Sample synthetic rows that satisfy user-supplied column conditions.

    Each condition dict must have the shape::

        {"num_rows": 1000, "column_values": {"medium": "paid_search"}}
    """
    synth = _load_synth(cfg, table_name)
    sdv_conditions = [
        Condition(num_rows=int(c["num_rows"]), column_values=dict(c["column_values"]))
        for c in conditions
    ]
    logger.info(
        "Conditional-sampling '%s' with %d condition group(s).",
        table_name,
        len(sdv_conditions),
    )
    df = synth.sample_from_conditions(conditions=sdv_conditions)
    logger.info("Produced %d conditional synthetic rows for '%s'.", len(df), table_name)
    return df

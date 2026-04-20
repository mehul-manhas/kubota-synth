"""Enforce cross-table business relationships after SDV sampling.

SDV synthesizes each table independently, so it cannot capture
business-level correlations such as "leads are ~0.3% of impressions on the
same day and channel" or "Ag equipment demand peaks in spring and fall".

This module reads a declarative YAML (``config/business_relationships.yaml``)
that encodes those rules and enforces them on the in-memory DataFrames right
before the writer stage runs. Every adjustment is logged with before/after
volumes so domain experts can audit what happened.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ConversionRate:
    mean: float
    std: float = 0.0

    def sample(self, rng: np.random.Generator) -> float:
        value = float(rng.normal(self.mean, self.std)) if self.std > 0 else float(self.mean)
        return max(0.0, min(1.0, value))


@dataclass
class DownstreamRule:
    table: str
    metric_column: str
    conversion_rate: ConversionRate
    join_on: list[str]


@dataclass
class FunnelChain:
    name: str
    driver_table: str
    driver_metric_column: str
    downstream: list[DownstreamRule] = field(default_factory=list)
    description: str = ""


@dataclass
class SeasonalityRule:
    name: str
    tables: list[str]
    metric_columns_per_table: dict[str, str]
    multipliers_by_month: dict[str, float]
    date_column: str = "date"
    date_columns_per_table: dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class BusinessRelationships:
    funnel_chains: list[FunnelChain] = field(default_factory=list)
    seasonality: list[SeasonalityRule] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.funnel_chains and not self.seasonality


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def load_business_relationships(path: str | Path) -> BusinessRelationships:
    """Load a business relationships YAML into typed dataclasses."""
    path = Path(path)
    if not path.exists():
        logger.info("No business relationships file at %s -- skipping post-processing.", path)
        return BusinessRelationships()

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}

    funnel_chains: list[FunnelChain] = []
    for chain_raw in raw.get("funnel_chains", []) or []:
        downstream: list[DownstreamRule] = []
        for ds in chain_raw.get("downstream", []) or []:
            cr = ds.get("conversion_rate", {}) or {}
            downstream.append(
                DownstreamRule(
                    table=ds["table"],
                    metric_column=ds["metric_column"],
                    conversion_rate=ConversionRate(
                        mean=float(cr.get("mean", 0.0)),
                        std=float(cr.get("std", 0.0)),
                    ),
                    join_on=list(ds.get("join_on", []) or []),
                )
            )
        funnel_chains.append(
            FunnelChain(
                name=chain_raw.get("name", "unnamed_chain"),
                driver_table=chain_raw["driver_table"],
                driver_metric_column=chain_raw["driver_metric_column"],
                downstream=downstream,
                description=str(chain_raw.get("description", "")).strip(),
            )
        )

    seasonality: list[SeasonalityRule] = []
    for season_raw in raw.get("seasonality", []) or []:
        multipliers_raw = season_raw.get("multipliers_by_month", {}) or {}
        multipliers = {str(k).zfill(2): float(v) for k, v in multipliers_raw.items()}
        seasonality.append(
            SeasonalityRule(
                name=season_raw.get("name", "unnamed_season"),
                tables=list(season_raw.get("tables", []) or []),
                metric_columns_per_table=dict(
                    season_raw.get("metric_columns_per_table", {}) or {}
                ),
                multipliers_by_month=multipliers,
                date_column=str(season_raw.get("date_column", "date")),
                date_columns_per_table=dict(
                    season_raw.get("date_columns_per_table", {}) or {}
                ),
                description=str(season_raw.get("description", "")).strip(),
            )
        )

    br = BusinessRelationships(funnel_chains=funnel_chains, seasonality=seasonality)
    logger.info(
        "Loaded business relationships: %d funnel chain(s), %d seasonality rule(s).",
        len(br.funnel_chains),
        len(br.seasonality),
    )
    return br


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_integer_series(series: pd.Series) -> bool:
    if pd.api.types.is_integer_dtype(series):
        return True
    # pandas nullable Int64 dtype
    if str(series.dtype).lower().startswith("int"):
        return True
    return False


def _coerce_back_to_original_dtype(original: pd.Series, updated: pd.Series) -> pd.Series:
    """Round/cast ``updated`` back to ``original``'s dtype when sensible."""
    if _is_integer_series(original):
        rounded = updated.round()
        try:
            return rounded.astype(original.dtype)
        except Exception:  # noqa: BLE001
            return rounded.astype("int64")
    return updated.astype(original.dtype, errors="ignore")


def _format_group_key(columns: list[str], key: Any) -> str:
    if isinstance(key, tuple):
        return ", ".join(f"{c}={v!r}" for c, v in zip(columns, key))
    if columns:
        return f"{columns[0]}={key!r}"
    return repr(key)


def _check_columns(df: pd.DataFrame, cols: list[str], table: str, kind: str) -> bool:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        logger.warning(
            "Skipping %s for table '%s': missing columns %s.", kind, table, missing
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Funnel chain enforcement
# ---------------------------------------------------------------------------


def apply_funnel_chain(
    tables: dict[str, pd.DataFrame],
    chain: FunnelChain,
    rng: np.random.Generator | None = None,
) -> dict[str, pd.DataFrame]:
    """Scale downstream-table metric columns to match driver * conversion rate.

    For each group (defined by the downstream rule's ``join_on`` columns) the
    target downstream volume is::

        target = sum(driver_metric in group) * sample(conversion_rate)

    and the downstream metric column is multiplied by ``target / current_sum``
    so the group totals line up.
    """
    if rng is None:
        rng = np.random.default_rng()

    if chain.driver_table not in tables:
        logger.warning(
            "Funnel chain '%s': driver table '%s' not in sampled tables -- skipping.",
            chain.name,
            chain.driver_table,
        )
        return tables

    driver_df = tables[chain.driver_table]
    if not _check_columns(
        driver_df, [chain.driver_metric_column], chain.driver_table, f"funnel '{chain.name}'"
    ):
        return tables

    for ds in chain.downstream:
        if ds.table not in tables:
            logger.info(
                "Funnel '%s': downstream table '%s' not sampled in this run -- skipping.",
                chain.name,
                ds.table,
            )
            continue

        child_df = tables[ds.table]
        if not _check_columns(
            driver_df, ds.join_on, chain.driver_table, f"funnel '{chain.name}' (driver join keys)"
        ):
            continue
        if not _check_columns(
            child_df, ds.join_on + [ds.metric_column], ds.table, f"funnel '{chain.name}'"
        ):
            continue

        driver_sums = (
            driver_df.groupby(ds.join_on, dropna=False)[chain.driver_metric_column].sum()
        )

        original_metric = child_df[ds.metric_column].copy()
        updated_metric = child_df[ds.metric_column].astype(float).copy()

        adjustments = 0
        zeroed = 0
        for group_key, child_group in child_df.groupby(ds.join_on, dropna=False):
            try:
                driver_sum = float(driver_sums.loc[group_key])
            except KeyError:
                driver_sum = 0.0

            rate = ds.conversion_rate.sample(rng)
            target_sum = driver_sum * rate
            current_sum = float(child_group[ds.metric_column].sum())
            idx = child_group.index

            if current_sum <= 0 and target_sum <= 0:
                continue

            if current_sum <= 0:
                # Evenly distribute target across rows in the group.
                if len(idx) > 0:
                    updated_metric.loc[idx] = target_sum / len(idx)
                    zeroed += 1
                continue

            scale = target_sum / current_sum
            updated_metric.loc[idx] = updated_metric.loc[idx] * scale
            adjustments += 1

            logger.info(
                "[%s -> %s] %s: scaled %s from %.2f to %.2f (rate=%.4f, scale=%.3f).",
                chain.driver_table,
                ds.table,
                _format_group_key(ds.join_on, group_key),
                ds.metric_column,
                current_sum,
                target_sum,
                rate,
                scale,
            )

        coerced = _coerce_back_to_original_dtype(original_metric, updated_metric)
        new_child_df = child_df.copy()
        new_child_df[ds.metric_column] = coerced
        tables[ds.table] = new_child_df

        logger.info(
            "Funnel '%s': adjusted %d group(s) in %s.%s (%d from-zero groups).",
            chain.name,
            adjustments,
            ds.table,
            ds.metric_column,
            zeroed,
        )

    return tables


# ---------------------------------------------------------------------------
# Seasonality enforcement
# ---------------------------------------------------------------------------


def apply_seasonality(
    tables: dict[str, pd.DataFrame],
    rule: SeasonalityRule,
) -> dict[str, pd.DataFrame]:
    """Multiply each table's metric column by the per-month multiplier."""
    for table in rule.tables:
        if table not in tables:
            logger.info(
                "Seasonality '%s': table '%s' not sampled in this run -- skipping.",
                rule.name,
                table,
            )
            continue

        df = tables[table]
        metric_col = rule.metric_columns_per_table.get(table)
        date_col = rule.date_columns_per_table.get(table, rule.date_column)

        if metric_col is None:
            logger.warning(
                "Seasonality '%s': no metric column configured for '%s' -- skipping.",
                rule.name,
                table,
            )
            continue

        if not _check_columns(df, [date_col, metric_col], table, f"seasonality '{rule.name}'"):
            continue

        dates = pd.to_datetime(df[date_col], errors="coerce")
        months = dates.dt.strftime("%m")
        multipliers = months.map(rule.multipliers_by_month).astype(float).fillna(1.0)

        original_metric = df[metric_col].copy()
        updated_metric = df[metric_col].astype(float) * multipliers
        coerced = _coerce_back_to_original_dtype(original_metric, updated_metric)

        new_df = df.copy()
        new_df[metric_col] = coerced
        tables[table] = new_df

        months_touched = multipliers[multipliers != 1.0].shape[0]
        before_sum = float(original_metric.sum())
        after_sum = float(updated_metric.sum())
        logger.info(
            "Seasonality '%s' on %s.%s: %d row(s) adjusted, total %.2f -> %.2f.",
            rule.name,
            table,
            metric_col,
            months_touched,
            before_sum,
            after_sum,
        )

    return tables


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def apply_relationships(
    tables: dict[str, pd.DataFrame],
    relationships: BusinessRelationships,
    seed: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Apply every funnel chain and seasonality rule, in that order.

    The ordering matters: funnel chains first tie downstream *volumes* to
    driver volumes, then seasonality scales everything by month-of-year.
    """
    if relationships.is_empty():
        logger.info("No business relationships to enforce.")
        return tables

    rng = np.random.default_rng(seed)

    for chain in relationships.funnel_chains:
        logger.info("Applying funnel chain '%s'.", chain.name)
        tables = apply_funnel_chain(tables, chain, rng=rng)

    for rule in relationships.seasonality:
        logger.info("Applying seasonality rule '%s'.", rule.name)
        tables = apply_seasonality(tables, rule)

    return tables

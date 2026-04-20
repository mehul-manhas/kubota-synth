"""Quality / diagnostic evaluation for synthetic data.

Wraps SDV's evaluation helpers and adds a few sanity checks geared towards
catching the most common synthesis bugs (all-null columns, wildly out-of-range
numerics, etc.).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic
from sdv.metadata import SingleTableMetadata

logger = logging.getLogger(__name__)


_NUMERIC_RANGE_TOLERANCE = 0.20


def _strip_internal_fields(metadata_dict: dict[str, Any]) -> dict[str, Any]:
    md = dict(metadata_dict or {})
    md.pop("_constraints", None)
    return md


def _sanity_issues(real: pd.DataFrame, synthetic: pd.DataFrame) -> list[str]:
    issues: list[str] = []

    for col in synthetic.columns:
        if col not in real.columns:
            continue

        real_series = real[col]
        synth_series = synthetic[col]

        # All-null synthetic values in a column that was mostly populated is suspicious.
        real_null_ratio = real_series.isna().mean() if len(real_series) else 1.0
        synth_null_ratio = synth_series.isna().mean() if len(synth_series) else 0.0
        if synth_null_ratio >= 0.99 and real_null_ratio < 0.5:
            issues.append(
                f"Column '{col}' is ~100% null in synthetic data but was "
                f"{real_null_ratio:.0%} null in real data."
            )

        if pd.api.types.is_numeric_dtype(real_series) and pd.api.types.is_numeric_dtype(
            synth_series
        ):
            real_min, real_max = real_series.min(), real_series.max()
            synth_min, synth_max = synth_series.min(), synth_series.max()
            if pd.isna(real_min) or pd.isna(real_max):
                continue
            span = float(real_max) - float(real_min)
            if span == 0:
                continue
            tolerance = span * _NUMERIC_RANGE_TOLERANCE

            if not pd.isna(synth_min) and float(synth_min) < float(real_min) - tolerance:
                issues.append(
                    f"Column '{col}' synthetic min ({synth_min}) is below real min "
                    f"({real_min}) by more than {_NUMERIC_RANGE_TOLERANCE:.0%} of range."
                )
            if not pd.isna(synth_max) and float(synth_max) > float(real_max) + tolerance:
                issues.append(
                    f"Column '{col}' synthetic max ({synth_max}) is above real max "
                    f"({real_max}) by more than {_NUMERIC_RANGE_TOLERANCE:.0%} of range."
                )

    return issues


def validate_table(
    table_name: str,
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    table_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Run SDV diagnostics + quality evaluation and return a summary dict."""
    md_dict = _strip_internal_fields(table_metadata)
    try:
        metadata = SingleTableMetadata.load_from_dict(md_dict)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not rebuild metadata for '%s': %s", table_name, exc)
        metadata = None

    diagnostic_score: float | None = None
    quality_score: float | None = None

    if metadata is not None and not real_df.empty and not synthetic_df.empty:
        try:
            diag = run_diagnostic(
                real_data=real_df,
                synthetic_data=synthetic_df,
                metadata=metadata,
                verbose=False,
            )
            diagnostic_score = float(diag.get_score())
        except Exception as exc:  # noqa: BLE001
            logger.warning("run_diagnostic failed for '%s': %s", table_name, exc)

        try:
            qual = evaluate_quality(
                real_data=real_df,
                synthetic_data=synthetic_df,
                metadata=metadata,
                verbose=False,
            )
            quality_score = float(qual.get_score())
        except Exception as exc:  # noqa: BLE001
            logger.warning("evaluate_quality failed for '%s': %s", table_name, exc)

    issues = _sanity_issues(real_df, synthetic_df)

    return {
        "table": table_name,
        "real_rows": int(len(real_df)),
        "synthetic_rows": int(len(synthetic_df)),
        "diagnostic_score": diagnostic_score,
        "quality_score": quality_score,
        "issues": issues,
    }


__all__ = ["validate_table"]


_ = np  # keep import for downstream users who extend this module

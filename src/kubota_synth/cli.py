"""Command-line interface for kubota-synth.

Commands
--------
* ``introspect`` -- read the source MSSQL schema and produce an SDV metadata JSON
* ``train``      -- train a synthesizer per table and persist it
* ``generate``   -- sample synthetic rows, optionally validate and write to MSSQL
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import click
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from kubota_synth.config import ProjectConfig, load_config
from kubota_synth.extract.data_loader import DataLoader
from kubota_synth.extract.introspect import MissingTablesError, build_sdv_metadata
from kubota_synth.load.mssql_writer import write_synthetic
from kubota_synth.postprocess.relationships import (
    apply_relationships,
    load_business_relationships,
)
from kubota_synth.synthesize.sampler import sample_table
from kubota_synth.synthesize.trainer import train_table
from kubota_synth.validate.quality import validate_table

DEFAULT_BUSINESS_RELATIONSHIPS_PATH = "config/business_relationships.yaml"

console = Console()
logger = logging.getLogger("kubota_synth")


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(numeric_level)
    root.addHandler(
        RichHandler(
            console=console,
            rich_tracebacks=True,
            show_path=False,
            omit_repeated_times=False,
        )
    )
    # Reduce SQLAlchemy spam unless user explicitly wants debug.
    if numeric_level > logging.DEBUG:
        logging.getLogger("sqlalchemy").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_project(config_path: str) -> ProjectConfig:
    try:
        return load_config(config_path)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Failed to load config:[/bold red] {exc}")
        sys.exit(1)


def _load_metadata(metadata_path: str) -> dict:
    path = Path(metadata_path)
    if not path.exists():
        console.print(
            f"[bold red]Metadata file not found:[/bold red] {path}\n"
            "Run `kubota-synth introspect -c <config> -o metadata.json` first."
        )
        sys.exit(1)
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_tables(cfg: ProjectConfig, table: str | None) -> list[str]:
    if table:
        if table not in cfg.tables:
            console.print(
                f"[bold red]Table '{table}' is not present in the config.[/bold red]"
            )
            sys.exit(1)
        return [table]
    return list(cfg.tables.keys())


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    show_default=True,
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    help="Logging verbosity.",
)
def cli(log_level: str) -> None:
    """kubota-synth: synthetic data generator for the Kubota platform."""
    _configure_logging(log_level)


# ---------------------------------------------------------------------------
# introspect
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "-c",
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to synthesizer_config.yaml.",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    default="metadata.json",
    show_default=True,
    type=click.Path(dir_okay=False),
    help="Where to write the SDV metadata JSON.",
)
def introspect(config_path: str, output_path: str) -> None:
    """Read the source MSSQL schema and emit an SDV metadata JSON."""
    cfg = _load_project(config_path)
    tables = list(cfg.tables.keys())
    if not tables:
        console.print("[bold red]No tables defined in the config.[/bold red]")
        sys.exit(1)

    console.print(
        f"[bold]Introspecting[/bold] {len(tables)} table(s) on "
        f"[cyan]{cfg.source.server}/{cfg.source.database}[/cyan] "
        f"(schema [cyan]{cfg.source.schema}[/cyan])..."
    )
    try:
        metadata = build_sdv_metadata(
            tables,
            cfg.source,
            schema=cfg.source.schema,
            overrides_dir=cfg.overrides_dir,
        )
    except MissingTablesError as exc:
        console.print(f"[bold red]Introspection failed:[/bold red] {exc}")
        console.print(
            "[dim]Fix the table names in your synthesizer config (or set "
            "SOURCE_SCHEMA in .env) and re-run.[/dim]"
        )
        sys.exit(2)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)

    console.print(f"[green]Wrote metadata for {len(tables)} table(s) -> {out}[/green]")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "-c",
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "-m",
    "--metadata",
    "metadata_path",
    required=True,
    type=click.Path(dir_okay=False),
    help="Path to the metadata JSON produced by `introspect`.",
)
@click.option(
    "--table",
    default=None,
    help="Train only this table (default: every table in the config).",
)
def train(config_path: str, metadata_path: str, table: str | None) -> None:
    """Train SDV synthesizers against the source database."""
    cfg = _load_project(config_path)
    metadata = _load_metadata(metadata_path)
    tables = _resolve_tables(cfg, table)

    summary_table = Table(title="Training summary")
    summary_table.add_column("Table", style="cyan")
    summary_table.add_column("Synthesizer")
    summary_table.add_column("Status")
    summary_table.add_column("Model path")

    for tbl in tables:
        table_cfg = cfg.tables[tbl]
        try:
            path = train_table(tbl, metadata, cfg)
            summary_table.add_row(tbl, table_cfg.synthesizer, "[green]ok[/green]", str(path))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Training failed for table '%s': %s", tbl, exc)
            summary_table.add_row(tbl, table_cfg.synthesizer, f"[red]failed: {exc}[/red]", "-")

    console.print(summary_table)


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@cli.command()
@click.option(
    "-c",
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "-m",
    "--metadata",
    "metadata_path",
    required=True,
    type=click.Path(dir_okay=False),
)
@click.option(
    "--table",
    default=None,
    help="Generate only this table (default: every table in the config).",
)
@click.option(
    "--rows",
    type=int,
    default=None,
    help="Override the per-table sample_rows setting.",
)
@click.option(
    "--write/--no-write",
    default=False,
    show_default=True,
    help="Write synthetic rows to the TARGET MSSQL database. Default is dry-run (CSV only).",
)
@click.option(
    "--output-dir",
    default=None,
    type=click.Path(file_okay=False),
    help="Directory for CSV output (default: artifacts/output).",
)
@click.option(
    "--validate",
    "do_validate",
    is_flag=True,
    default=False,
    help="Run SDV quality/diagnostic scoring against fresh real data.",
)
@click.option(
    "--apply-relationships/--no-apply-relationships",
    "apply_rel",
    default=True,
    show_default=True,
    help=(
        "Enforce cross-table business rules (funnel conversion, seasonality) "
        "defined in config/business_relationships.yaml."
    ),
)
@click.option(
    "--relationships-config",
    "relationships_path",
    default=None,
    type=click.Path(dir_okay=False),
    help=(
        "Path to the business-relationships YAML. "
        f"Default: {DEFAULT_BUSINESS_RELATIONSHIPS_PATH} (if present)."
    ),
)
def generate(
    config_path: str,
    metadata_path: str,
    table: str | None,
    rows: int | None,
    write: bool,
    output_dir: str | None,
    do_validate: bool,
    apply_rel: bool,
    relationships_path: str | None,
) -> None:
    """Sample synthetic rows, enforce business rules, validate, optionally write."""
    cfg = _load_project(config_path)
    metadata = _load_metadata(metadata_path)
    tables = _resolve_tables(cfg, table)

    out_dir = Path(output_dir) if output_dir else cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not write:
        console.print(
            "[yellow]DRY RUN:[/yellow] generating CSVs only. "
            "Pass [bold]--write[/bold] to append to the target MSSQL database."
        )

    # -- 1. Sample every table into memory -------------------------------------
    sampled: dict[str, pd.DataFrame] = {}
    errors: dict[str, str] = {}
    for tbl in tables:
        table_cfg = cfg.tables[tbl]
        target_rows = rows if rows is not None else table_cfg.sample_rows
        try:
            sampled[tbl] = sample_table(tbl, cfg, rows=target_rows)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Sampling failed for table '%s': %s", tbl, exc)
            errors[tbl] = str(exc)

    # -- 2. Apply business-relationship rules across tables --------------------
    relationships_applied = False
    relationships_summary: str = "disabled"
    if apply_rel:
        rel_path = Path(relationships_path) if relationships_path else Path(
            DEFAULT_BUSINESS_RELATIONSHIPS_PATH
        )
        if rel_path.exists():
            relationships = load_business_relationships(rel_path)
            if relationships.is_empty():
                relationships_summary = "empty-config"
                console.print(
                    f"[yellow]Relationships config {rel_path} is empty -- skipping.[/yellow]"
                )
            else:
                seed = int(cfg.defaults.get("random_seed", 42))
                console.print(
                    f"[bold]Applying business relationships[/bold] from [cyan]{rel_path}[/cyan]..."
                )
                sampled = apply_relationships(sampled, relationships, seed=seed)
                relationships_applied = True
                relationships_summary = (
                    f"{len(relationships.funnel_chains)}f/{len(relationships.seasonality)}s"
                )
        else:
            console.print(
                f"[yellow]No relationships config at {rel_path} -- skipping post-processing."
                "[/yellow]"
            )
            relationships_summary = "no-config"
    else:
        console.print("[yellow]--no-apply-relationships: skipping post-processing.[/yellow]")

    # -- 3. Per-table: CSV snapshot, validation, optional DB write ------------
    summary_table = Table(title="Generation summary")
    summary_table.add_column("Table", style="cyan")
    summary_table.add_column("Generated", justify="right")
    summary_table.add_column("Written", justify="right")
    summary_table.add_column("Quality", justify="right")
    summary_table.add_column("Diagnostic", justify="right")
    summary_table.add_column("Issues", justify="right")
    summary_table.add_column("Rules", justify="center")
    summary_table.add_column("CSV")

    rules_cell = "[green]applied[/green]" if relationships_applied else (
        "[dim]skipped[/dim]" if relationships_summary in {"disabled", "no-config", "empty-config"}
        else "-"
    )

    for tbl in tables:
        generated = 0
        written = 0
        quality_display = "-"
        diagnostic_display = "-"
        issues_display = "-"
        csv_path = out_dir / f"{tbl}.csv"

        if tbl in errors:
            summary_table.add_row(
                tbl, "0", "0", "-", "-", "-", rules_cell,
                f"[red]failed: {errors[tbl]}[/red]",
            )
            continue

        df = sampled.get(tbl)
        if df is None:
            summary_table.add_row(
                tbl, "0", "0", "-", "-", "-", rules_cell, "[red]no sample[/red]",
            )
            continue

        try:
            generated = len(df)
            df.to_csv(csv_path, index=False)
            logger.info("Wrote CSV sample for '%s' -> %s", tbl, csv_path)

            if do_validate:
                report = _validate_against_real(tbl, df, metadata, cfg)
                if report is not None:
                    if report["quality_score"] is not None:
                        quality_display = f"{report['quality_score']:.3f}"
                    if report["diagnostic_score"] is not None:
                        diagnostic_display = f"{report['diagnostic_score']:.3f}"
                    issues_display = str(len(report["issues"]))
                    for issue in report["issues"]:
                        logger.warning("[%s] %s", tbl, issue)

            if write:
                written = write_synthetic(tbl, df, cfg)

            summary_table.add_row(
                tbl,
                str(generated),
                str(written),
                quality_display,
                diagnostic_display,
                issues_display,
                rules_cell,
                str(csv_path),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Post-sampling pipeline failed for table '%s': %s", tbl, exc)
            summary_table.add_row(
                tbl,
                str(generated),
                str(written),
                quality_display,
                diagnostic_display,
                issues_display,
                rules_cell,
                f"[red]failed: {exc}[/red]",
            )

    console.print(summary_table)
    console.print(f"[dim]Business relationships: {relationships_summary}[/dim]")


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------


def _validate_against_real(
    table_name: str,
    synthetic_df: pd.DataFrame,
    metadata: dict,
    cfg: ProjectConfig,
) -> dict | None:
    """Fetch a fresh real sample and run ``validate_table`` against it."""
    tables_meta = metadata.get("tables", {}) if isinstance(metadata, dict) else {}
    table_metadata = tables_meta.get(table_name)
    if table_metadata is None:
        logger.warning("No metadata entry for '%s' -- skipping validation.", table_name)
        return None

    sample_size = max(len(synthetic_df), 10_000)
    try:
        with DataLoader(cfg.source, schema=cfg.source.schema) as loader:
            real_df = loader.load(table_name, sample_size=sample_size)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load real data for '%s' validation: %s", table_name, exc)
        return None

    return validate_table(table_name, real_df, synthetic_df, table_metadata)


if __name__ == "__main__":  # pragma: no cover
    cli()

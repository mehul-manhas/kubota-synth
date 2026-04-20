# kubota-synth

Synthetic data generator for the Kubota marketing analytics platform. Reads
real data from MSSQL (`sql-kubota-dev`), trains [SDV] synthesizers on it, and
writes synthetic rows to a copy database (`sql-kubota-dev_copy`) so the full
bronze → silver → gold pipeline can be exercised end-to-end.

[SDV]: https://docs.sdv.dev/sdv

## Why this exists

- Some periods (months) in the real marketing data have missing or low-volume
  records. We need to backfill those gaps with statistically plausible rows.
- Dashboard and QA teams need realistic volumes we do not have in production.
- The silver-to-gold transformation layer already exists and runs on top of
  the bronze layer — we just need a way to populate bronze (and a handful of
  silver tables) with synthetic data that keeps the downstream models happy.

## Flow

```
 sql-kubota-dev                                                          sql-kubota-dev_copy
      |                                                                          ^
      | introspect      train       sample     enforce-rules   validate  write   |
      +-----------> metadata.json -> *.pkl --> per-table df -> funnel/-> quality->+
                    (SDV V1)        models    (in memory)     seasonality scores
```

1. `introspect` reads `INFORMATION_SCHEMA` + `sys.foreign_keys` and writes
   `metadata.json` (SDV V1 spec).
2. `train` pulls a sampled slice of each source table, fits the configured
   synthesizer (GaussianCopula / CTGAN / TVAE / PAR) and persists it as a
   `.pkl`.
3. `generate` samples every configured table into memory, then:
   - applies cross-table **business relationships** (funnel conversion rates,
     seasonality multipliers) from `config/business_relationships.yaml`,
   - always dumps a CSV per table into `artifacts/output/`,
   - optionally runs `run_diagnostic` + `evaluate_quality` against a fresh
     real sample (`--validate`),
   - optionally appends rows to the target database via `fast_executemany`
     bulk-insert (`--write`).

## Setup

```bash
# 1. Create a virtual environment for Python 3.10–3.12
python -m venv .venv
source .venv/bin/activate

# 2. Install the package (+ dev tools if you plan to hack on it)
pip install -e ".[dev]"

# 3. Copy the env template and fill in credentials
cp .env.example .env
$EDITOR .env

# 4. Edit the config to point at real tables (see below)
$EDITOR config/synthesizer_config.yaml
```

Requires the Microsoft ODBC 18 driver installed locally (macOS: `brew install
msodbcsql18`; Linux: see Microsoft's docs). On-prem / offline friendly — no
external API calls.

## Usage

All three commands take `-c`/`--config` pointing at `synthesizer_config.yaml`.
Add `--log-level DEBUG` to the `kubota-synth` group for verbose tracing.

### 1. Introspect the source schema

```bash
kubota-synth introspect \
  -c config/synthesizer_config.yaml \
  -o metadata.json
```

Writes an SDV V1 metadata JSON. Every column gets a best-guess `sdtype`:

| MSSQL type                                                    | sdtype        |
|---------------------------------------------------------------|---------------|
| `int`, `bigint`, `decimal`, `numeric`, `float`, `money`, ...  | `numerical`   |
| `datetime`, `datetime2`, `date`, `time`, `datetimeoffset`     | `datetime`    |
| `varchar`, `nvarchar`, `char`, `nchar` (≤200 distinct values) | `categorical` |
| `varchar`, `nvarchar`, `char`, `nchar` (>200 distinct values) | `text`        |
| `text`, `ntext`                                               | `text`        |
| `bit`                                                         | `boolean`     |
| `uniqueidentifier` or the PK column                           | `id`          |

Drop a `config/overrides/<table>.yaml` file to override any column or add
SDV constraints (see `config/overrides/_example.yaml`).

### 2. Train the synthesizers

```bash
kubota-synth train \
  -c config/synthesizer_config.yaml \
  -m metadata.json

# Just one table
kubota-synth train -c config/synthesizer_config.yaml -m metadata.json --table bronze_campaigns
```

Models land in `artifacts/models/<table>.pkl`. The trainer honours the
per-table `fit_sample_size` and uses MSSQL's `TABLESAMPLE` for efficient
sampling on large tables.

### 3. Generate synthetic rows

```bash
# Dry run: CSVs only, no DB writes
kubota-synth generate -c config/synthesizer_config.yaml -m metadata.json

# With SDV quality scoring against a fresh real sample
kubota-synth generate -c config/synthesizer_config.yaml -m metadata.json --validate

# Actually append to sql-kubota-dev_copy
kubota-synth generate -c config/synthesizer_config.yaml -m metadata.json --validate --write

# Skip cross-table business rules (leave each table as SDV sampled it)
kubota-synth generate -c config/synthesizer_config.yaml -m metadata.json --no-apply-relationships

# Use a custom relationships config
kubota-synth generate -c config/synthesizer_config.yaml -m metadata.json \
  --relationships-config config/backfill_q4.yaml

# Single table with custom row count
kubota-synth generate -c config/synthesizer_config.yaml -m metadata.json \
  --table bronze_placements --rows 250000 --write
```

`generate` is **dry-run by default** — it will only write CSVs into
`artifacts/output/` unless you pass `--write`. Business-relationship rules
(funnel conversion, seasonality) are applied by default when
`config/business_relationships.yaml` exists; pass
`--no-apply-relationships` to disable. The end-of-run summary table shows
rows generated, rows written, quality/diagnostic scores, issue count, and
whether business rules were applied.

## Synthesizer cheat sheet

| Synthesizer      | Best for                                                   | Training cost            | Quality (vs. real)                |
|------------------|------------------------------------------------------------|--------------------------|-----------------------------------|
| `GaussianCopula` | Small/medium tables, mostly numeric + low-card categorical | Very fast (seconds)      | Good for distributions, weak on joint structure |
| `CTGAN`          | Wide tables with complex categorical interactions          | Slow on CPU, needs GPU   | Best for realistic joint distributions |
| `TVAE`           | Similar to CTGAN, lighter-weight                           | Moderate                 | Slightly worse than CTGAN, faster |
| `PAR`            | Sequential / time-ordered data (events, placements)        | Slow                     | Preserves temporal structure — requires `sequence_key` |

PAR also accepts an optional `sequence_index` (e.g. a date column) to order
sequences chronologically.

## Business relationships (cross-table rules)

SDV models each table independently, which means it cannot learn
relationships like "leads are ~0.3% of impressions on the same day and
channel" or "ag equipment demand peaks in spring and fall." These are
**business facts**, not statistical ones — you know them, SDV doesn't.

`kubota-synth` encodes them in a declarative YAML file that runs as a
post-processing stage **after** sampling and **before** validation/write.
The default path is `config/business_relationships.yaml`. Override with
`--relationships-config PATH` or disable entirely with
`--no-apply-relationships`.

### Funnel chains

A funnel chain declares a **driver table** and one or more **downstream
tables**. For each `join_on` group (e.g. same date + channel) the downstream
metric column is scaled so that

```
sum(downstream_metric in group)  =  sum(driver_metric in group) * sampled_rate
```

where `sampled_rate` is drawn per group from `Normal(mean, std)` clipped to
`[0, 1]`. This lets you say things like "for any given day/channel, leads
are roughly 0.3% ± 0.1% of impressions."

```yaml
funnel_chains:
  - name: paid_media_funnel
    driver_table: bronze_impressions
    driver_metric_column: impression_count
    downstream:
      - table: bronze_leads
        metric_column: lead_count
        conversion_rate: {mean: 0.003, std: 0.001}
        join_on: [date, channel]
      - table: bronze_quotes
        metric_column: quote_count
        conversion_rate: {mean: 0.15, std: 0.05}
        join_on: [date, channel]
```

### Seasonality

Multiply a metric column by a per-month coefficient. Month is derived from
the configured `date_column` (default `date`).

```yaml
seasonality:
  - name: ag_equipment_cycle
    tables: [bronze_impressions, bronze_leads, bronze_quotes]
    date_column: date
    metric_columns_per_table:
      bronze_impressions: impression_count
      bronze_leads: lead_count
      bronze_quotes: quote_count
    multipliers_by_month:
      "03": 1.3
      "04": 1.4
      "05": 1.3
      "09": 1.2
      "10": 1.3
      "12": 0.7
      "01": 0.7
      # any month left out defaults to 1.0 (no change)
```

### Ordering and auditability

Rules run funnel-chains-first, then seasonality, so the seasonal multiplier
is applied *on top of* the funnel-scaled downstream volume. Every single
adjustment is logged at INFO level, e.g.

```
[bronze_impressions -> bronze_leads] date='2025-10-01', channel='paid_search':
  scaled lead_count from 412.00 to 98.40 (rate=0.0031, scale=0.239)
Seasonality 'ag_equipment_cycle' on bronze_leads.lead_count:
  1284 row(s) adjusted, total 98400.00 -> 127920.00
```

so domain experts can eyeball the deltas and decide whether the configured
rates are right.

### When to use which

| Problem                                                       | Use              |
|---------------------------------------------------------------|------------------|
| "These tables share a date/channel grain and should correlate" | funnel_chains   |
| "Volumes should rise in spring, drop in winter"                | seasonality     |
| "Campaign has many placements has many impressions (FK-level)" | HMASynthesizer (future work — not implemented) |
| "Two rows must satisfy col_a < col_b"                          | per-table SDV constraints in `config/overrides/<table>.yaml` |

Integer metric columns stay integer (scaling is rounded back to the column's
original dtype).

## Creating target tables

`kubota-synth` appends — it never creates tables. Before your first `--write`
run, create shape-only copies in the target database. For cross-server copies:

```sql
SELECT TOP 0 *
INTO dbo.bronze_placements
FROM [sql-kubota-dev].kubota.dbo.bronze_placements;

SELECT TOP 0 *
INTO dbo.bronze_impressions
FROM [sql-kubota-dev].kubota.dbo.bronze_impressions;

-- ... etc for every table listed in synthesizer_config.yaml
```

You need a linked server or equivalent to resolve `[sql-kubota-dev]`. If
that's not set up, run the `SELECT TOP 0 * INTO ... FROM ...` statement while
connected to the source, generate a DDL script from it, and execute it on the
target.

## YAML override reference

Place a file named `<table_name>.yaml` in `config/overrides/`. Example:

```yaml
columns:
  placement_id:
    sdtype: id
  placement_date:
    sdtype: datetime
    datetime_format: "%Y-%m-%d"
  landing_url:
    sdtype: text

constraints:
  - constraint_class: Positive
    constraint_parameters:
      column_name: impressions
      strict_boundaries: false
  - constraint_class: Inequality
    constraint_parameters:
      low_column_name: start_date
      high_column_name: end_date
```

Any SDV-supported constraint works (`Positive`, `Negative`, `ScalarRange`,
`ScalarInequality`, `Inequality`, `FixedCombinations`, `OneHotEncoding`, ...).

## Known constraints and gotchas

- **Composite primary keys are not supported** by SDV OSS. `kubota-synth`
  logs a warning and falls back to the first PK column.
- **Cross-table FK integrity is not preserved.** Each table is synthesized
  independently, so synthetic PKs are new and will not match synthetic FKs
  in a referential-integrity sense. If you need that, model the tables
  together using the SDV `HMASynthesizer` (future work).
- **Business correlations across tables are handled by post-processing, not
  by SDV.** SDV is statistical, not semantic — it cannot infer "leads are
  ~0.3% of impressions" from column names. Encode those rules in
  `config/business_relationships.yaml` (see the Business relationships
  section above).
- **CTGAN / TVAE are slow on CPU.** On wide tables (>50 columns) expect tens
  of minutes per fit on a reasonable laptop. Use `fit_sample_size` to cap
  training rows.
- **Datetime parsing is strict.** The introspector sets
  `datetime_format: "%Y-%m-%d %H:%M:%S"` by default. Override in
  `config/overrides/<table>.yaml` if your data uses a different format.
- **Quality evaluation is best-effort.** If SDV's evaluation fails (it
  occasionally does on unusual schemas) we log a warning and continue with
  the generation — synthesis is not blocked by validation.
- **`--write` is opt-in.** By design, you cannot accidentally mutate the
  target database without passing `--write`. The default is CSV-only.
- **`fast_executemany=True`** on the target engine is critical for bulk
  inserts. Keep it enabled.

## Development

```bash
pip install -e ".[dev]"
ruff check src
pytest
```

## Project layout

```
kubota-synth/
├── pyproject.toml
├── README.md
├── .env.example
├── .gitignore
├── config/
│   ├── synthesizer_config.yaml     # tables + synthesizer choice + defaults
│   ├── business_relationships.yaml # cross-table funnel + seasonality rules
│   └── overrides/                  # per-table sdtype / constraint overrides
├── src/kubota_synth/
│   ├── cli.py                      # Click CLI entrypoint
│   ├── config.py                   # dataclasses + YAML/env loader
│   ├── extract/                    # MSSQL introspection + data loading
│   ├── synthesize/                 # SDV registry, trainer, sampler
│   ├── postprocess/                # business-rules enforcement
│   ├── validate/                   # SDV quality/diagnostic wrappers
│   └── load/                       # MSSQL bulk writer
├── tests/
└── artifacts/
    ├── models/                     # trained *.pkl files
    └── output/                     # CSV snapshots of generated data
```

## License

MIT.

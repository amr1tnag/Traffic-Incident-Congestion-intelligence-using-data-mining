"""Stage 8 — export the warehouse and the mining results for Power BI.

Power BI Desktop has no native SQLite connector, so the star schema is written
out as UTF-8 CSV (one file per table) into a single folder that Power Query
points at.  The mining outputs travel with them as satellite tables, which is
what lets the Power BI report show clusters, forecasts and anomalies without
re-implementing any model in DAX.

    python -m src.export_bi                  # writes powerbi/data/
    python -m src.export_bi --out some/dir
"""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT, load_config

DEFAULT_OUT = PROJECT_ROOT / "powerbi" / "data"

# Warehouse tables exported verbatim.  The agg_* cuboids come along because they
# make the OLAP page's roll-up visuals cheap.
DW_TABLES = [
    "dim_time", "dim_segment", "dim_weather", "dim_incident_type",
    "fact_traffic", "fact_incident",
    "agg_zone_month", "agg_roadtype_hour", "agg_segment_daily",
]

# Mining CSVs copied straight across: reports/<source> -> <target>.csv
MINING_CSVS = {
    "segment_clusters.csv": "mining_segment_clusters",
    "congestion_forecast.csv": "mining_forecast",
    "anomalies.csv": "mining_anomalies",
    "association_rules.csv": "mining_association_rules",
}

# Boolean-ish integer columns, flagged so the semantic model can type them.
FLAG_COLUMNS = {
    "dim_time": ["is_weekend", "is_peak_hour"],
    "dim_segment": ["signalised"],
    "dim_weather": ["is_adverse"],
    "fact_traffic": ["is_congested"],
    "fact_incident": ["is_major"],
}


def build_dim_date(dim_time: pd.DataFrame) -> pd.DataFrame:
    """Derive a day-grain date dimension from the hourly time dimension.

    Power BI can only mark a table as *the* date table when its date column is
    unique and contiguous, which the hour-grain dim_time is not.  dim_date sits
    above it (dim_date -> dim_time -> facts) so DAX time intelligence works
    while the hourly grain stays available for the congestion-profile visuals.
    """
    dates = pd.to_datetime(dim_time["date"]).drop_duplicates().sort_values()
    full = pd.date_range(dates.min(), dates.max(), freq="D")  # contiguous
    dim = pd.DataFrame({"date": full})
    dim.insert(0, "date_key", dim["date"].dt.strftime("%Y%m%d").astype("int64"))
    dim["year"] = dim["date"].dt.year
    dim["quarter"] = dim["date"].dt.quarter
    dim["quarter_name"] = "Q" + dim["quarter"].astype(str)
    dim["month"] = dim["date"].dt.month
    dim["month_name"] = dim["date"].dt.strftime("%B")
    dim["year_month"] = dim["date"].dt.strftime("%Y-%m")
    dim["week_of_year"] = dim["date"].dt.isocalendar().week.astype("int64")
    dim["day_of_month"] = dim["date"].dt.day
    dim["day_of_week"] = dim["date"].dt.dayofweek + 1
    dim["day_name"] = dim["date"].dt.strftime("%A")
    dim["is_weekend"] = (dim["date"].dt.dayofweek >= 5).astype(int)
    dim["date"] = dim["date"].dt.strftime("%Y-%m-%d")
    return dim


def export_warehouse(db_path: Path, out_dir: Path) -> dict[str, int]:
    """Dump every star-schema table to CSV.  Returns table -> row count."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"warehouse not found at {db_path} — run `python run_pipeline.py` first")

    counts: dict[str, int] = {}
    with sqlite3.connect(db_path) as conn:
        present = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
        for table in DW_TABLES:
            if table not in present:
                print(f"  ! {table:<22} missing from the warehouse, skipped")
                continue
            frame = pd.read_sql_query(f"SELECT * FROM {table}", conn)
            if table == "dim_time":
                # Foreign key up to dim_date, so the day-grain dimension can
                # filter everything the hour-grain dimension filters.
                frame.insert(1, "date_key",
                             pd.to_datetime(frame["date"]).dt.strftime("%Y%m%d").astype("int64"))
                dim_date = build_dim_date(frame)
                _write(dim_date, out_dir / "dim_date.csv")
                counts["dim_date"] = len(dim_date)
                print(f"  + {'dim_date':<22} {len(dim_date):>8,} rows")
            _write(frame, out_dir / f"{table}.csv")
            counts[table] = len(frame)
            print(f"  + {table:<22} {len(frame):>8,} rows")
    return counts


def export_mining(reports_dir: Path, out_dir: Path) -> dict[str, int]:
    """Copy the per-technique result tables next to the warehouse extract."""
    counts: dict[str, int] = {}
    for source, target in MINING_CSVS.items():
        path = reports_dir / source
        if not path.exists():
            print(f"  ! {source:<28} not produced yet, skipped")
            continue
        frame = pd.read_csv(path)
        _write(frame, out_dir / f"{target}.csv")
        counts[target] = len(frame)
        print(f"  + {target:<22} {len(frame):>8,} rows")
    return counts


def export_model_metrics(reports_dir: Path, out_dir: Path) -> int:
    """Flatten the *_results.json metrics into one tall table for the KPI page.

    One row per (task, model, metric) so a single matrix visual can compare
    every model without a column per metric.
    """
    rows: list[dict[str, object]] = []

    classification = _load_json(reports_dir / "classification_results.json")
    for model, metrics in (classification.get("models") or {}).items():
        for metric in ("cv_accuracy_mean", "test_accuracy",
                       "macro_precision", "macro_recall", "macro_f1"):
            if metric in metrics:
                rows.append({"task": "Classification", "model": model,
                             "metric": metric, "value": metrics[metric]})

    forecasting = _load_json(reports_dir / "forecasting_results.json")
    for model, metrics in (forecasting.get("metrics") or {}).items():
        for metric, value in metrics.items():
            rows.append({"task": "Forecasting", "model": model,
                         "metric": metric, "value": value})

    clustering = _load_json(reports_dir / "clustering_results.json")
    for diagnostic in clustering.get("k_diagnostics") or []:
        for metric in ("silhouette", "davies_bouldin", "calinski_harabasz", "inertia"):
            if metric in diagnostic:
                rows.append({"task": "Clustering", "model": f"k={diagnostic['k']}",
                             "metric": metric, "value": diagnostic[metric]})

    anomaly = _load_json(reports_dir / "anomaly_results.json")
    for metric in ("rows_scored", "iso_anomalies", "zscore_anomalies",
                   "agreed_anomalies", "agreed_without_logged_incident"):
        if metric in anomaly:
            rows.append({"task": "Anomaly detection", "model": "ensemble",
                         "metric": metric, "value": anomaly[metric]})

    association = _load_json(reports_dir / "association_results.json")
    for metric in ("n_transactions", "n_rules"):
        if metric in association:
            rows.append({"task": "Association rules", "model": "apriori",
                         "metric": metric, "value": association[metric]})

    frame = pd.DataFrame(rows, columns=["task", "model", "metric", "value"])
    _write(frame, out_dir / "mining_model_metrics.csv")
    print(f"  + mining_model_metrics  {len(frame):>8,} rows")
    return len(frame)


def export_figures(figures_dir: Path, out_dir: Path) -> int:
    """Copy the matplotlib figures so report pages can embed them as images."""
    if not figures_dir.exists():
        return 0
    target = out_dir / "figures"
    target.mkdir(parents=True, exist_ok=True)
    pngs = sorted(figures_dir.glob("*.png"))
    for png in pngs:
        shutil.copy2(png, target / png.name)
    if pngs:
        print(f"  + figures/              {len(pngs):>8,} images")
    return len(pngs)


def write_manifest(out_dir: Path, counts: dict[str, int]) -> Path:
    """Record what was exported, so a stale refresh is obvious in the report."""
    manifest = pd.DataFrame(
        sorted(counts.items()), columns=["table", "row_count"])
    manifest["exported_at"] = pd.Timestamp.now().floor("s").isoformat(sep=" ")
    path = out_dir / "export_manifest.csv"
    _write(manifest, path)
    return path


def _write(frame: pd.DataFrame, path: Path) -> None:
    # utf-8-sig: Power Query's CSV connector reads the BOM and stops guessing
    # the encoding, which otherwise mangles non-ASCII segment names.
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def main(config_path: str | None = None, out_dir: Path | str | None = None) -> Path:
    cfg = load_config(config_path)
    out = Path(out_dir) if out_dir else DEFAULT_OUT
    out.mkdir(parents=True, exist_ok=True)
    reports_dir = Path(cfg["reports"]["dir"])

    print(f"Exporting to {out}")
    counts = export_warehouse(Path(cfg["warehouse"]["path"]), out)
    counts |= export_mining(reports_dir, out)
    counts["mining_model_metrics"] = export_model_metrics(reports_dir, out)
    export_figures(Path(cfg["reports"]["figures_dir"]), out)
    write_manifest(out, counts)

    print(f"\n{len(counts)} tables written. In Power BI Desktop open "
          f"powerbi/TrafficIntelligence.pbip and set the DataFolder parameter to:\n  {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None, help="target folder (default powerbi/data)")
    args = ap.parse_args()
    main(args.config, args.out)

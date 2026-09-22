"""Stage 2 — data cleaning, integration, transformation and reduction.

This is the classic KDD preprocessing chain applied to the raw feeds:

  Cleaning      : duplicates, impossible sensor values, missing-value imputation,
                  unification of inconsistent categorical encodings.
  Integration   : readings + incidents + road-segment reference data are joined
                  on (segment_id, hour) into one analytical base table.
  Transformation: calendar/cyclical features, ratios, discretisation (binning)
                  of the continuous congestion index into ordered classes.
  Reduction     : hourly granularity is kept for forecasting, while a compact
                  incident-level table is produced for classification.

Every step records how many rows/values it touched; the counts land in
``reports/preprocessing_report.json`` and are worth quoting in the write-up.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config

VALID_RANGES = {
    "avg_speed_kmph": (0.0, 160.0),
    "occupancy_pct": (0.0, 100.0),
    "vehicle_count": (0.0, 20000.0),
    "temperature_c": (-10.0, 55.0),
    "precipitation_mm": (0.0, 200.0),
}
PEAK_HOURS = {7, 8, 9, 17, 18, 19}


def _normalise_weather(series: pd.Series) -> pd.Series:
    """Two upstream vendors disagree on casing/spelling; fold them together."""
    cleaned = series.astype("object").where(series.notna(), None)
    cleaned = cleaned.map(lambda w: " ".join(str(w).strip().split()).title() if w is not None else None)
    return cleaned.replace({"Heavyrain": "Heavy Rain", "None": None, "Nan": None})


def clean_readings(readings: pd.DataFrame, outlier_z: float) -> tuple[pd.DataFrame, dict]:
    log: dict[str, int | float] = {"rows_in": int(len(readings))}
    df = readings.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    before = len(df)
    df = df.drop_duplicates(subset=["segment_id", "timestamp"], keep="first")
    log["duplicate_rows_removed"] = int(before - len(df))

    # Out-of-range sensor readings are not real observations — void them first,
    # then impute, so a 999999 vehicle count never pollutes a mean.
    voided = 0
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            bad = ~df[col].between(lo, hi) & df[col].notna()
            voided += int(bad.sum())
            df.loc[bad, col] = np.nan
    log["out_of_range_values_voided"] = voided

    df["weather"] = _normalise_weather(df["weather"])
    log["weather_missing_before"] = int(df["weather"].isna().sum())
    # Weather is a city-wide property of the hour: fill from the same timestamp.
    hour_mode = (
        df.dropna(subset=["weather"]).groupby("timestamp")["weather"]
        .agg(lambda s: s.mode().iat[0])
    )
    df["weather"] = df["weather"].fillna(df["timestamp"].map(hour_mode)).fillna("Clear")

    numeric = ["vehicle_count", "avg_speed_kmph", "occupancy_pct", "travel_time_min",
               "congestion_index", "temperature_c", "precipitation_mm"]
    log["missing_values_imputed"] = int(df[numeric].isna().sum().sum())
    # Missing telemetry is imputed from the same segment at the same hour of the
    # week — the most similar observations available — then by column median.
    keys = [df["segment_id"], df["timestamp"].dt.dayofweek, df["timestamp"].dt.hour]
    for col in numeric:
        if df[col].isna().any():
            df[col] = df[col].fillna(df.groupby(keys)[col].transform("median"))
            df[col] = df[col].fillna(df[col].median())

    # Residual extreme values (kept, but flagged) — a z-score screen per segment.
    z = df.groupby("segment_id")["congestion_index"].transform(
        lambda s: (s - s.mean()) / (s.std(ddof=0) or 1.0)
    )
    df["is_outlier"] = (z.abs() > outlier_z).astype(int)
    log["outliers_flagged"] = int(df["is_outlier"].sum())

    df = df.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)
    log["rows_out"] = int(len(df))
    return df, log


def clean_incidents(incidents: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    log: dict[str, int] = {"rows_in": int(len(incidents))}
    df = incidents.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    before = len(df)
    df = df.drop_duplicates(subset=["incident_id"])
    log["duplicate_rows_removed"] = int(before - len(df))

    df["weather"] = _normalise_weather(df["weather"]).fillna("Clear")
    log["missing_values_imputed"] = int(df[["duration_min", "response_time_min"]].isna().sum().sum())
    # Duration/response gaps are filled from peers of the same type and severity.
    for col in ["duration_min", "response_time_min"]:
        df[col] = df[col].fillna(df.groupby(["incident_type", "severity"])[col].transform("median"))
        df[col] = df[col].fillna(df[col].median())

    df["severity"] = df["severity"].astype(int).clip(1, 4)
    log["rows_out"] = int(len(df))
    return df, log


def add_time_features(df: pd.DataFrame, ts_col: str = "timestamp") -> pd.DataFrame:
    """Calendar attributes plus cyclical encodings (hour 23 is next to hour 0)."""
    ts = df[ts_col]
    out = df.copy()
    out["date"] = ts.dt.date.astype(str)
    out["year"] = ts.dt.year
    out["quarter"] = ts.dt.quarter
    out["month"] = ts.dt.month
    out["month_name"] = ts.dt.month_name()
    out["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    out["day_of_month"] = ts.dt.day
    out["day_of_week"] = ts.dt.dayofweek
    out["day_name"] = ts.dt.day_name()
    out["hour"] = ts.dt.hour
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    out["is_peak_hour"] = out["hour"].isin(PEAK_HOURS).astype(int)
    out["time_of_day"] = pd.cut(
        out["hour"], bins=[-1, 5, 11, 16, 20, 23],
        labels=["Night", "Morning", "Afternoon", "Evening", "Late Evening"],
    ).astype(str)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    return out


def discretise_congestion(df: pd.DataFrame, bins: list[float], labels: list[str]) -> pd.DataFrame:
    """Equal-width binning of the continuous index into ordered traffic states."""
    out = df.copy()
    out["congestion_level"] = pd.cut(
        out["congestion_index"], bins=bins, labels=labels, include_lowest=True
    ).astype(str)
    out["is_congested"] = (out["congestion_index"] >= bins[2]).astype(int)
    return out


def build_analytical_base(
    readings: pd.DataFrame, incidents: pd.DataFrame, segments: pd.DataFrame
) -> pd.DataFrame:
    """Integration step: one row per segment-hour with network context attached."""
    inc_hour = incidents.assign(hour_bucket=incidents["timestamp"].dt.floor("h"))
    per_hour = inc_hour.groupby(["segment_id", "hour_bucket"]).agg(
        incident_count=("incident_id", "count"),
        max_severity=("severity", "max"),
        total_lanes_blocked=("lanes_blocked", "sum"),
        mean_response_time=("response_time_min", "mean"),
    ).reset_index()

    df = readings.merge(segments, on="segment_id", how="left")
    df = df.merge(
        per_hour, left_on=["segment_id", "timestamp"],
        right_on=["segment_id", "hour_bucket"], how="left",
    ).drop(columns=["hour_bucket"])

    for col, fill in [("incident_count", 0), ("max_severity", 0),
                      ("total_lanes_blocked", 0), ("mean_response_time", 0.0)]:
        df[col] = df[col].fillna(fill)
    df["incident_count"] = df["incident_count"].astype(int)
    df["max_severity"] = df["max_severity"].astype(int)

    df["speed_ratio"] = (df["avg_speed_kmph"] / df["speed_limit_kmph"]).clip(0, 1.5).round(4)
    df["delay_index"] = (1 - df["speed_ratio"]).clip(0, 1).round(4)
    df["flow_per_lane"] = (df["vehicle_count"] / df["lanes"]).round(2)
    df["vehicle_km"] = (df["vehicle_count"] * df["length_km"]).round(2)
    df["is_wet"] = df["weather"].isin(["Rain", "Heavy Rain"]).astype(int)
    return df


def build_incident_features(incidents: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Incident-level mining table: the target is ``severity``.

    Each incident is enriched with the traffic state of its own segment-hour,
    which is what a control room would actually have at decision time.
    """
    context_cols = [
        "segment_id", "timestamp", "congestion_index", "congestion_level", "avg_speed_kmph",
        "speed_ratio", "occupancy_pct", "flow_per_lane", "vehicle_count",
        "road_type", "zone", "lanes", "length_km", "speed_limit_kmph", "signalised",
        "hour", "day_of_week", "day_name", "month", "is_weekend", "is_peak_hour", "time_of_day",
    ]
    ctx = base[context_cols].copy()
    df = incidents.assign(hour_bucket=incidents["timestamp"].dt.floor("h")).merge(
        ctx, left_on=["segment_id", "hour_bucket"], right_on=["segment_id", "timestamp"],
        how="left", suffixes=("", "_ctx"),
    ).drop(columns=["hour_bucket", "timestamp_ctx"])

    df["blocked_lane_ratio"] = (df["lanes_blocked"] / df["lanes"].clip(lower=1)).round(3)
    df["severity_label"] = df["severity"].map(
        {1: "Minor", 2: "Moderate", 3: "Serious", 4: "Critical"}
    )
    df["is_major"] = (df["severity"] >= 3).astype(int)
    df["duration_band"] = pd.cut(
        df["duration_min"], bins=[-1, 20, 45, 90, 10_000],
        labels=["<20m", "20-45m", "45-90m", ">90m"],
    ).astype(str)
    df["response_band"] = pd.cut(
        df["response_time_min"], bins=[-1, 8, 15, 25, 10_000],
        labels=["Fast", "Normal", "Slow", "Very Slow"],
    ).astype(str)
    return df.dropna(subset=["congestion_index"]).reset_index(drop=True)


def main(config_path: str | None = None) -> dict[str, Path]:
    cfg = load_config(config_path)
    raw, proc = Path(cfg["data"]["raw_dir"]), Path(cfg["data"]["processed_dir"])

    segments = pd.read_csv(raw / "road_segments.csv")
    readings = pd.read_csv(raw / "traffic_readings.csv")
    incidents = pd.read_csv(raw / "incidents.csv")

    readings, rlog = clean_readings(readings, cfg["preprocess"]["outlier_z"])
    incidents, ilog = clean_incidents(incidents)

    readings = add_time_features(readings)
    readings = discretise_congestion(
        readings, cfg["preprocess"]["congestion_bins"], cfg["preprocess"]["congestion_labels"]
    )
    base = build_analytical_base(readings, incidents, segments)
    incidents = add_time_features(incidents)
    incident_features = build_incident_features(incidents, base)

    paths = {
        "segments": proc / "dim_segments.csv",
        "base": proc / "analytical_base.csv",
        "incidents": proc / "incidents_clean.csv",
        "incident_features": proc / "incident_features.csv",
    }
    segments.to_csv(paths["segments"], index=False)
    base.to_csv(paths["base"], index=False)
    incidents.to_csv(paths["incidents"], index=False)
    incident_features.to_csv(paths["incident_features"], index=False)

    report = {"readings": rlog, "incidents": ilog,
              "analytical_base_rows": int(len(base)),
              "incident_feature_rows": int(len(incident_features))}
    out = Path(cfg["reports"]["dir"]) / "preprocessing_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[preprocess] duplicates removed   : {rlog['duplicate_rows_removed']:,}")
    print(f"[preprocess] impossible values    : {rlog['out_of_range_values_voided']:,}")
    print(f"[preprocess] values imputed       : {rlog['missing_values_imputed'] + ilog['missing_values_imputed']:,}")
    print(f"[preprocess] outliers flagged     : {rlog['outliers_flagged']:,}")
    print(f"[preprocess] analytical base rows : {len(base):,}")
    print(f"[preprocess] incident mining rows : {len(incident_features):,}")
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Clean, integrate and transform the raw feeds.")
    ap.add_argument("--config", default=None)
    main(ap.parse_args().config)

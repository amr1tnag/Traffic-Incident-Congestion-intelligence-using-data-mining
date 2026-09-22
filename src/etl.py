"""Stage 3 — load the cleaned data into the star-schema warehouse (SQLite).

Surrogate keys are generated for every dimension, the facts are keyed against
them, and a handful of materialised aggregate views are created so the OLAP
layer answers roll-up questions without rescanning the base fact table.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from .config import load_config

SCHEMA_FILE = Path(__file__).with_name("warehouse_schema.sql")
ADVERSE_WEATHER = {"Rain", "Heavy Rain", "Fog"}
INCIDENT_CATEGORY = {
    "Collision": "Accident", "Pedestrian": "Accident",
    "Breakdown": "Obstruction", "Debris": "Obstruction", "Signal Failure": "Obstruction",
    "Road Work": "Planned",
}

TIME_COLS = ["timestamp", "date", "year", "quarter", "month", "month_name", "week_of_year",
             "day_of_month", "day_of_week", "day_name", "hour", "time_of_day",
             "is_weekend", "is_peak_hour"]


def time_key(ts: pd.Series) -> pd.Series:
    """Readable smart key: yyyymmddhh."""
    ts = pd.to_datetime(ts)
    return (ts.dt.strftime("%Y%m%d%H")).astype("int64")


def build_dim_time(base: pd.DataFrame, incidents: pd.DataFrame) -> pd.DataFrame:
    frames = [base[TIME_COLS], incidents[TIME_COLS]]
    dim = pd.concat(frames, ignore_index=True)
    dim["timestamp"] = pd.to_datetime(dim["timestamp"]).dt.floor("h").astype(str)
    dim = dim.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    dim.insert(0, "time_key", time_key(dim["timestamp"]))
    return dim.drop_duplicates(subset=["time_key"]).reset_index(drop=True)


def build_dim_segment(segments: pd.DataFrame) -> pd.DataFrame:
    dim = segments.copy().sort_values("segment_id").reset_index(drop=True)
    dim.insert(0, "segment_key", range(1, len(dim) + 1))
    return dim


def build_dim_weather(*frames: pd.DataFrame) -> pd.DataFrame:
    values = sorted({w for f in frames for w in f["weather"].dropna().unique()})
    return pd.DataFrame({
        "weather_key": range(1, len(values) + 1),
        "weather": values,
        "is_adverse": [int(w in ADVERSE_WEATHER) for w in values],
    })


def build_dim_incident_type(incidents: pd.DataFrame) -> pd.DataFrame:
    values = sorted(incidents["incident_type"].dropna().unique())
    return pd.DataFrame({
        "incident_type_key": range(1, len(values) + 1),
        "incident_type": values,
        "category": [INCIDENT_CATEGORY.get(v, "Other") for v in values],
    })


def build_fact_traffic(base: pd.DataFrame, dim_segment: pd.DataFrame,
                       dim_weather: pd.DataFrame) -> pd.DataFrame:
    df = base.copy()
    df["time_key"] = time_key(df["timestamp"])
    df = df.merge(dim_segment[["segment_key", "segment_id"]], on="segment_id", how="left")
    df = df.merge(dim_weather[["weather_key", "weather"]], on="weather", how="left")
    cols = ["time_key", "segment_key", "weather_key", "vehicle_count", "avg_speed_kmph",
            "occupancy_pct", "travel_time_min", "congestion_index", "congestion_level",
            "speed_ratio", "delay_index", "flow_per_lane", "vehicle_km",
            "incident_count", "is_congested"]
    fact = df[cols].reset_index(drop=True)
    fact.insert(0, "traffic_key", range(1, len(fact) + 1))
    return fact


def build_fact_incident(incidents: pd.DataFrame, base: pd.DataFrame, dim_segment: pd.DataFrame,
                        dim_weather: pd.DataFrame, dim_type: pd.DataFrame) -> pd.DataFrame:
    df = incidents.copy()
    df["time_key"] = time_key(df["timestamp"])

    ctx = base[["segment_id", "timestamp", "congestion_index"]].copy()
    ctx["time_key"] = time_key(ctx["timestamp"])
    df = df.merge(ctx[["segment_id", "time_key", "congestion_index"]],
                  on=["segment_id", "time_key"], how="left")

    df = df.merge(dim_segment[["segment_key", "segment_id"]], on="segment_id", how="left")
    df = df.merge(dim_weather[["weather_key", "weather"]], on="weather", how="left")
    df = df.merge(dim_type[["incident_type_key", "incident_type"]], on="incident_type", how="left")
    df["is_major"] = (df["severity"] >= 3).astype(int)

    cols = ["incident_id", "time_key", "segment_key", "weather_key", "incident_type_key",
            "severity", "vehicles_involved", "lanes_blocked", "casualties",
            "duration_min", "response_time_min", "congestion_index", "is_major"]
    fact = df[cols].reset_index(drop=True)
    fact.insert(0, "incident_key", range(1, len(fact) + 1))
    return fact


AGGREGATE_VIEWS = {
    # Pre-aggregated cuboids of the traffic cube — the classic materialised
    # summary tables that make roll-up queries cheap.
    "agg_zone_month": """
        CREATE TABLE agg_zone_month AS
        SELECT s.zone, t.year, t.month, t.month_name,
               COUNT(*)                       AS observations,
               ROUND(AVG(f.congestion_index),4) AS avg_congestion,
               ROUND(AVG(f.avg_speed_kmph),2)   AS avg_speed,
               SUM(f.incident_count)            AS incidents,
               ROUND(AVG(f.is_congested)*100,2) AS pct_congested_hours
        FROM fact_traffic f
        JOIN dim_segment s ON s.segment_key = f.segment_key
        JOIN dim_time    t ON t.time_key    = f.time_key
        GROUP BY s.zone, t.year, t.month, t.month_name
    """,
    "agg_roadtype_hour": """
        CREATE TABLE agg_roadtype_hour AS
        SELECT s.road_type, t.hour, t.is_weekend,
               ROUND(AVG(f.congestion_index),4) AS avg_congestion,
               ROUND(AVG(f.delay_index),4)      AS avg_delay_index,
               COUNT(*)                         AS observations
        FROM fact_traffic f
        JOIN dim_segment s ON s.segment_key = f.segment_key
        JOIN dim_time    t ON t.time_key    = f.time_key
        GROUP BY s.road_type, t.hour, t.is_weekend
    """,
    "agg_segment_daily": """
        CREATE TABLE agg_segment_daily AS
        SELECT s.segment_id, s.zone, s.road_type, t.date,
               ROUND(AVG(f.congestion_index),4) AS avg_congestion,
               MAX(f.congestion_index)          AS peak_congestion,
               SUM(f.incident_count)            AS incidents,
               ROUND(SUM(f.vehicle_km),2)       AS vehicle_km
        FROM fact_traffic f
        JOIN dim_segment s ON s.segment_key = f.segment_key
        JOIN dim_time    t ON t.time_key    = f.time_key
        GROUP BY s.segment_id, s.zone, s.road_type, t.date
    """,
}


def load_warehouse(cfg: dict) -> Path:
    proc = Path(cfg["data"]["processed_dir"])
    db_path = Path(cfg["warehouse"]["path"])
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    segments = pd.read_csv(proc / "dim_segments.csv")
    base = pd.read_csv(proc / "analytical_base.csv")
    incidents = pd.read_csv(proc / "incidents_clean.csv")

    dim_time = build_dim_time(base, incidents)
    dim_segment = build_dim_segment(segments)
    dim_weather = build_dim_weather(base, incidents)
    dim_type = build_dim_incident_type(incidents)
    fact_traffic = build_fact_traffic(base, dim_segment, dim_weather)
    fact_incident = build_fact_incident(incidents, base, dim_segment, dim_weather, dim_type)

    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_FILE.read_text(encoding="utf-8"))
        for name, frame in [
            ("dim_time", dim_time), ("dim_segment", dim_segment),
            ("dim_weather", dim_weather), ("dim_incident_type", dim_type),
            ("fact_traffic", fact_traffic), ("fact_incident", fact_incident),
        ]:
            frame.to_sql(name, conn, if_exists="append", index=False)
            print(f"[etl] loaded {name:<18} {len(frame):>8,} rows")
        for name, sql in AGGREGATE_VIEWS.items():
            conn.execute(f"DROP TABLE IF EXISTS {name}")
            conn.execute(sql)
            print(f"[etl] materialised {name}")
        conn.commit()
    print(f"[etl] warehouse ready at {db_path}")
    return db_path


def main(config_path: str | None = None) -> Path:
    return load_warehouse(load_config(config_path))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Load the star-schema data warehouse.")
    ap.add_argument("--config", default=None)
    main(ap.parse_args().config)

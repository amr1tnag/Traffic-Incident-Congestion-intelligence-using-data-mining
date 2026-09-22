"""Stage 1 — build the raw operational data sources.

The project is designed around three source systems that a city traffic authority
would realistically own:

  * ``road_segments.csv``   — a master/reference table of the road network.
  * ``traffic_readings.csv``— hourly loop-detector + weather telemetry.
  * ``incidents.csv``       — the incident log raised by police / control room.

A public feed for a specific city is not always reachable from an offline lab
machine, so the readings are simulated from documented traffic behaviour
(morning/evening peaks, weekday-weekend split, weather penalties, incident
shockwaves). Realistic data-quality defects are injected on purpose so the
preprocessing stage has genuine work to do.

Drop real CSVs with the same column names into ``data/raw/`` and the rest of the
pipeline runs unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_config

ROAD_TYPES = ["Highway", "Arterial", "Collector", "Residential"]
ROAD_TYPE_WEIGHTS = [0.15, 0.35, 0.30, 0.20]
ZONES = ["North", "South", "East", "West", "Central"]
WEATHER = ["Clear", "Cloudy", "Rain", "Heavy Rain", "Fog"]
WEATHER_P = [0.52, 0.24, 0.14, 0.05, 0.05]
INCIDENT_TYPES = ["Collision", "Breakdown", "Road Work", "Debris", "Signal Failure", "Pedestrian"]
INCIDENT_P = [0.34, 0.24, 0.16, 0.12, 0.08, 0.06]

# Multiplicative demand profile by hour of day (0-23): twin commuter peaks.
HOUR_PROFILE = np.array([
    0.18, 0.12, 0.09, 0.09, 0.14, 0.30, 0.58, 0.86, 0.98, 0.82, 0.68, 0.66,
    0.70, 0.68, 0.66, 0.72, 0.88, 1.00, 0.95, 0.74, 0.56, 0.44, 0.34, 0.24,
])
# Multiplicative demand profile by day of week (Mon=0 .. Sun=6).
DOW_PROFILE = np.array([1.00, 1.02, 1.03, 1.04, 1.08, 0.82, 0.62])
WEATHER_PENALTY = {"Clear": 0.0, "Cloudy": 0.01, "Rain": 0.07, "Heavy Rain": 0.14, "Fog": 0.10}
ROAD_TYPE_CAPACITY = {"Highway": 1900, "Arterial": 1100, "Collector": 700, "Residential": 380}


def build_segments(rng: np.random.Generator, n_segments: int) -> pd.DataFrame:
    """Reference data for the road network (a dimension in warehouse terms)."""
    road_type = rng.choice(ROAD_TYPES, size=n_segments, p=ROAD_TYPE_WEIGHTS)
    lanes = np.where(
        road_type == "Highway", rng.integers(3, 6, n_segments),
        np.where(road_type == "Arterial", rng.integers(2, 4, n_segments),
                 rng.integers(1, 3, n_segments)),
    )
    speed_limit = pd.Series(road_type).map(
        {"Highway": 100, "Arterial": 60, "Collector": 50, "Residential": 30}
    ).to_numpy()
    return pd.DataFrame({
        "segment_id": [f"SEG{ i :03d}" for i in range(1, n_segments + 1)],
        "segment_name": [f"{z} Link {i}" for i, z in enumerate(rng.choice(ZONES, n_segments), 1)],
        "zone": rng.choice(ZONES, n_segments),
        "road_type": road_type,
        "lanes": lanes,
        "length_km": np.round(rng.uniform(0.4, 6.5, n_segments), 2),
        "speed_limit_kmph": speed_limit,
        "signalised": rng.choice([0, 1], n_segments, p=[0.45, 0.55]),
        "latitude": np.round(12.90 + rng.uniform(-0.18, 0.18, n_segments), 5),
        "longitude": np.round(77.59 + rng.uniform(-0.18, 0.18, n_segments), 5),
    })


def _hourly_index(start_date: str, n_days: int) -> pd.DatetimeIndex:
    return pd.date_range(start=start_date, periods=n_days * 24, freq="h")


def build_readings_and_incidents(
    rng: np.random.Generator,
    segments: pd.DataFrame,
    start_date: str,
    n_days: int,
    incident_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate hourly telemetry, raising incidents that feed back into congestion."""
    stamps = _hourly_index(start_date, n_days)
    n_hours = len(stamps)
    hour_of_day = stamps.hour.to_numpy()
    dow = stamps.dayofweek.to_numpy()

    # One weather state per hour for the whole city, with light persistence.
    weather = rng.choice(WEATHER, size=n_hours, p=WEATHER_P)
    carry = rng.random(n_hours) < 0.55
    for i in range(1, n_hours):
        if carry[i]:
            weather[i] = weather[i - 1]
    temperature = 24 + 7 * np.sin((np.arange(n_hours) - 2400) / 1400.0) \
        + 4 * np.sin((hour_of_day - 9) / 24 * 2 * np.pi) + rng.normal(0, 1.2, n_hours)
    precip = np.where(
        np.isin(weather, ["Rain", "Heavy Rain"]),
        rng.gamma(2.0, 2.5, n_hours) * np.where(weather == "Heavy Rain", 3.0, 1.0),
        0.0,
    )

    reading_rows, incident_rows = [], []
    incident_seq = 0

    for seg in segments.itertuples(index=False):
        capacity = ROAD_TYPE_CAPACITY[seg.road_type] * seg.lanes
        # Each segment has its own popularity and a slow yearly growth trend.
        popularity = rng.uniform(0.55, 1.15)
        trend = np.linspace(1.0, rng.uniform(1.0, 1.12), n_hours)

        demand = capacity * popularity * HOUR_PROFILE[hour_of_day] * DOW_PROFILE[dow] * trend
        demand *= rng.normal(1.0, 0.09, n_hours).clip(0.6, 1.5)

        weather_pen = np.array([WEATHER_PENALTY[w] for w in weather])

        # --- incidents ---------------------------------------------------
        load_ratio = np.clip(demand / capacity, 0, 2.0)
        hazard = incident_rate * (0.35 + 0.9 * load_ratio) * (1 + 3.0 * weather_pen)
        hazard *= {"Highway": 1.25, "Arterial": 1.0, "Collector": 0.7, "Residential": 0.45}[seg.road_type]
        struck = rng.random(n_hours) < np.clip(hazard, 0, 0.6)
        struck_idx = np.flatnonzero(struck)

        blockage = np.zeros(n_hours)
        for idx in struck_idx:
            incident_seq += 1
            itype = rng.choice(INCIDENT_TYPES, p=INCIDENT_P)
            vehicles = int(np.clip(rng.poisson(1.6) + (2 if itype == "Collision" else 1), 1, 9))
            lanes_blocked = int(np.clip(rng.integers(0, seg.lanes + 1), 0, seg.lanes))
            # Severity is a latent score driven by the same factors an analyst
            # would expect, then discretised into the 1-4 scale used by police.
            score = (
                1.15 * lanes_blocked / max(seg.lanes, 1)
                + 0.42 * (vehicles - 1) / 8
                + 0.65 * weather_pen[idx] / 0.14
                + 0.55 * load_ratio[idx]
                + 0.50 * (itype == "Collision")
                + 0.30 * (seg.road_type == "Highway")
                + 0.25 * (hour_of_day[idx] in (7, 8, 17, 18))
                + rng.normal(0, 0.30)
            )
            severity = int(np.clip(np.digitize(score, [0.85, 1.55, 2.25]) + 1, 1, 4))
            duration = float(np.clip(rng.gamma(2.2, 9 + 11 * severity), 5, 360))
            response = float(np.clip(rng.gamma(2.0, 3.0 + 1.6 * (5 - severity)), 2, 75))

            incident_rows.append({
                "incident_id": f"INC{incident_seq:06d}",
                "segment_id": seg.segment_id,
                "timestamp": stamps[idx],
                "incident_type": itype,
                "severity": severity,
                "vehicles_involved": vehicles,
                "lanes_blocked": lanes_blocked,
                "duration_min": round(duration, 1),
                "response_time_min": round(response, 1),
                "weather": weather[idx],
                "casualties": int(rng.poisson(0.25 * severity)) if itype == "Collision" else 0,
            })

            # An incident degrades capacity for as long as it lasts.
            span = max(1, int(np.ceil(duration / 60)))
            hit = 0.20 + 0.55 * (lanes_blocked / max(seg.lanes, 1)) + 0.10 * severity / 4
            for k in range(span):
                if idx + k < n_hours:
                    blockage[idx + k] = max(blockage[idx + k], hit * (0.65 ** k))

        # --- congestion ---------------------------------------------------
        effective_capacity = capacity * (1 - np.clip(blockage, 0, 0.9))
        congestion = np.clip(demand / np.maximum(effective_capacity, 1) * 0.75 + weather_pen, 0.02, 1.6)
        congestion = np.clip(congestion + rng.normal(0, 0.03, n_hours), 0.02, 1.0)

        free_flow = seg.speed_limit_kmph * 0.95
        avg_speed = np.clip(free_flow * (1 - 0.82 * congestion) + rng.normal(0, 2.0, n_hours), 3, free_flow)
        vehicle_count = np.round(np.minimum(demand, effective_capacity) * rng.uniform(0.92, 1.0)).astype(int)
        occupancy = np.clip(congestion * 100 * rng.normal(1.0, 0.05, n_hours), 0.5, 99.5)
        travel_time = seg.length_km / np.maximum(avg_speed, 1) * 60

        reading_rows.append(pd.DataFrame({
            "segment_id": seg.segment_id,
            "timestamp": stamps,
            "vehicle_count": vehicle_count,
            "avg_speed_kmph": np.round(avg_speed, 2),
            "occupancy_pct": np.round(occupancy, 2),
            "travel_time_min": np.round(travel_time, 3),
            "congestion_index": np.round(congestion, 4),
            "weather": weather,
            "temperature_c": np.round(temperature, 1),
            "precipitation_mm": np.round(precip, 2),
            "incident_active": (blockage > 0).astype(int),
        }))

    readings = pd.concat(reading_rows, ignore_index=True)
    readings.insert(0, "reading_id", [f"RD{i:08d}" for i in range(1, len(readings) + 1)])
    incidents = pd.DataFrame(incident_rows).sort_values("timestamp").reset_index(drop=True)
    return readings, incidents


def inject_quality_issues(
    rng: np.random.Generator, readings: pd.DataFrame, incidents: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Dirty the data the way a real feed is dirty, so cleaning is a real step."""
    readings = readings.copy()
    incidents = incidents.copy()
    n = len(readings)

    # Sensor dropouts -> missing measurements.
    for col, frac in [("avg_speed_kmph", 0.018), ("occupancy_pct", 0.012), ("vehicle_count", 0.008)]:
        readings.loc[rng.choice(n, int(n * frac), replace=False), col] = np.nan
    readings.loc[rng.choice(n, int(n * 0.02), replace=False), "weather"] = None

    # Miscalibrated detectors -> impossible values.
    readings.loc[rng.choice(n, int(n * 0.004), replace=False), "avg_speed_kmph"] = -1.0
    readings.loc[rng.choice(n, int(n * 0.003), replace=False), "occupancy_pct"] = 150.0
    readings.loc[rng.choice(n, int(n * 0.002), replace=False), "vehicle_count"] = 999999

    # Inconsistent categorical encoding from two upstream vendors.
    flip = rng.choice(n, int(n * 0.06), replace=False)
    readings.loc[flip, "weather"] = readings.loc[flip, "weather"].astype("object").map(
        lambda w: str(w).upper() if isinstance(w, str) else w
    )

    # Duplicate transmissions.
    dupes = readings.sample(frac=0.01, random_state=int(rng.integers(1e6)))
    readings = pd.concat([readings, dupes], ignore_index=True)

    # Incident log gaps.
    m = len(incidents)
    incidents.loc[rng.choice(m, int(m * 0.03), replace=False), "response_time_min"] = np.nan
    incidents.loc[rng.choice(m, int(m * 0.02), replace=False), "duration_min"] = np.nan
    return readings, incidents


def main(config_path: str | None = None) -> dict[str, Path]:
    cfg = load_config(config_path)
    rng = np.random.default_rng(cfg["seed"])
    d = cfg["data"]
    raw_dir = Path(d["raw_dir"])

    segments = build_segments(rng, d["n_segments"])
    readings, incidents = build_readings_and_incidents(
        rng, segments, d["start_date"], d["n_days"], d["incident_rate"]
    )
    readings, incidents = inject_quality_issues(rng, readings, incidents)

    paths = {
        "segments": raw_dir / "road_segments.csv",
        "readings": raw_dir / "traffic_readings.csv",
        "incidents": raw_dir / "incidents.csv",
    }
    segments.to_csv(paths["segments"], index=False)
    readings.to_csv(paths["readings"], index=False)
    incidents.to_csv(paths["incidents"], index=False)

    print(f"[generate] segments : {len(segments):>8,} rows -> {paths['segments'].name}")
    print(f"[generate] readings : {len(readings):>8,} rows -> {paths['readings'].name}")
    print(f"[generate] incidents: {len(incidents):>8,} rows -> {paths['incidents'].name}")
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate the raw traffic data sources.")
    ap.add_argument("--config", default=None)
    main(ap.parse_args().config)

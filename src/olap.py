"""Stage 4 — OLAP layer over the warehouse.

Demonstrates the five canonical cube operations against the traffic cube
(dimensions: Time x Location x Weather x Incident Type; measures: congestion
index, speed, delay, incident counts, duration):

    roll_up      zone-month summary climbing the segment -> zone -> city hierarchy
    drill_down   one zone broken back down to segment and hour
    slice        fix one dimension (e.g. weather = 'Heavy Rain')
    dice         a sub-cube on several dimension ranges at once
    pivot        reorient the cube as road_type x hour
    cube         multi-level aggregation with GROUPING SETS semantics

Results are written to ``reports/olap/`` as CSV so they can be pasted into the
project report.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

from .config import load_config


class TrafficCube:
    """Thin query facade over the star schema."""

    def __init__(self, db_path: str | Path):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row

    def __enter__(self) -> "TrafficCube":
        return self

    def __exit__(self, *exc) -> None:
        self.conn.close()

    def q(self, sql: str, params: tuple = ()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn, params=params)

    # ------------------------------------------------------------ roll-up
    def roll_up_zone_month(self) -> pd.DataFrame:
        """Climb the location hierarchy to zone and the time hierarchy to month."""
        return self.q("""
            SELECT zone, year, month, month_name, observations,
                   avg_congestion, avg_speed, incidents, pct_congested_hours
            FROM agg_zone_month
            ORDER BY zone, year, month
        """)

    def roll_up_city_quarter(self) -> pd.DataFrame:
        """Top of both hierarchies: the whole city by quarter."""
        return self.q("""
            SELECT t.year, t.quarter,
                   ROUND(AVG(f.congestion_index),4) AS avg_congestion,
                   ROUND(AVG(f.avg_speed_kmph),2)   AS avg_speed,
                   SUM(f.incident_count)            AS incidents,
                   ROUND(SUM(f.vehicle_km),0)       AS vehicle_km
            FROM fact_traffic f JOIN dim_time t ON t.time_key = f.time_key
            GROUP BY t.year, t.quarter ORDER BY t.year, t.quarter
        """)

    # ---------------------------------------------------------- drill-down
    def drill_down_zone(self, zone: str) -> pd.DataFrame:
        """Descend from a zone to its individual segments and hours."""
        return self.q("""
            SELECT s.zone, s.segment_id, s.road_type, t.hour,
                   ROUND(AVG(f.congestion_index),4) AS avg_congestion,
                   ROUND(AVG(f.avg_speed_kmph),2)   AS avg_speed,
                   SUM(f.incident_count)            AS incidents
            FROM fact_traffic f
            JOIN dim_segment s ON s.segment_key = f.segment_key
            JOIN dim_time    t ON t.time_key    = f.time_key
            WHERE s.zone = ?
            GROUP BY s.zone, s.segment_id, s.road_type, t.hour
            ORDER BY avg_congestion DESC
        """, (zone,))

    # --------------------------------------------------------------- slice
    def slice_by_weather(self, weather: str) -> pd.DataFrame:
        """Fix the weather dimension to a single value."""
        return self.q("""
            SELECT w.weather, s.road_type,
                   COUNT(*)                         AS observations,
                   ROUND(AVG(f.congestion_index),4) AS avg_congestion,
                   ROUND(AVG(f.delay_index),4)      AS avg_delay_index,
                   ROUND(AVG(f.is_congested)*100,2) AS pct_congested_hours
            FROM fact_traffic f
            JOIN dim_weather w ON w.weather_key = f.weather_key
            JOIN dim_segment s ON s.segment_key = f.segment_key
            WHERE w.weather = ?
            GROUP BY w.weather, s.road_type ORDER BY avg_congestion DESC
        """, (weather,))

    # ---------------------------------------------------------------- dice
    def dice(self, zones: list[str], road_types: list[str], hours: tuple[int, int]) -> pd.DataFrame:
        """A sub-cube constrained on three dimensions at once."""
        zq = ",".join("?" * len(zones))
        rq = ",".join("?" * len(road_types))
        return self.q(f"""
            SELECT s.zone, s.road_type, t.day_name,
                   ROUND(AVG(f.congestion_index),4) AS avg_congestion,
                   ROUND(AVG(f.avg_speed_kmph),2)   AS avg_speed,
                   SUM(f.incident_count)            AS incidents
            FROM fact_traffic f
            JOIN dim_segment s ON s.segment_key = f.segment_key
            JOIN dim_time    t ON t.time_key    = f.time_key
            WHERE s.zone IN ({zq}) AND s.road_type IN ({rq}) AND t.hour BETWEEN ? AND ?
            GROUP BY s.zone, s.road_type, t.day_name
            ORDER BY avg_congestion DESC
        """, (*zones, *road_types, hours[0], hours[1]))

    # --------------------------------------------------------------- pivot
    def pivot_roadtype_hour(self) -> pd.DataFrame:
        """Reorient the cube: rows = road type, columns = hour of day."""
        raw = self.q("""
            SELECT road_type, hour, ROUND(AVG(avg_congestion),4) AS avg_congestion
            FROM agg_roadtype_hour GROUP BY road_type, hour
        """)
        return raw.pivot(index="road_type", columns="hour", values="avg_congestion").round(3)

    def pivot_severity_by_weather(self) -> pd.DataFrame:
        raw = self.q("""
            SELECT w.weather, i.severity, COUNT(*) AS incidents
            FROM fact_incident i JOIN dim_weather w ON w.weather_key = i.weather_key
            GROUP BY w.weather, i.severity
        """)
        return raw.pivot(index="weather", columns="severity", values="incidents").fillna(0).astype(int)

    # ---------------------------------------------------------------- cube
    def cube_incidents(self) -> pd.DataFrame:
        """Multi-level aggregation: (type x zone), (type), (zone) and the grand
        total — SQLite has no GROUPING SETS, so the cuboids are unioned."""
        levels = [
            ("SELECT it.incident_type AS incident_type, s.zone AS zone,", "GROUP BY it.incident_type, s.zone"),
            ("SELECT it.incident_type AS incident_type, 'ALL' AS zone,", "GROUP BY it.incident_type"),
            ("SELECT 'ALL' AS incident_type, s.zone AS zone,", "GROUP BY s.zone"),
            ("SELECT 'ALL' AS incident_type, 'ALL' AS zone,", ""),
        ]
        body = """
               COUNT(*) AS incidents,
               ROUND(AVG(i.severity),3)          AS avg_severity,
               ROUND(AVG(i.duration_min),2)      AS avg_duration_min,
               ROUND(AVG(i.response_time_min),2) AS avg_response_min,
               SUM(i.casualties)                 AS casualties
        FROM fact_incident i
        JOIN dim_segment s       ON s.segment_key       = i.segment_key
        JOIN dim_incident_type it ON it.incident_type_key = i.incident_type_key
        """
        sql = "\nUNION ALL\n".join(f"{head}{body}{tail}" for head, tail in levels)
        return self.q(sql).sort_values(["incident_type", "zone"]).reset_index(drop=True)

    # ------------------------------------------------- analytical highlights
    def top_congested_segments(self, n: int = 10) -> pd.DataFrame:
        return self.q("""
            SELECT s.segment_id, s.zone, s.road_type,
                   ROUND(AVG(f.congestion_index),4)   AS avg_congestion,
                   ROUND(AVG(f.is_congested)*100,2)   AS pct_congested_hours,
                   SUM(f.incident_count)              AS incidents
            FROM fact_traffic f JOIN dim_segment s ON s.segment_key = f.segment_key
            GROUP BY s.segment_id, s.zone, s.road_type
            ORDER BY avg_congestion DESC LIMIT ?
        """, (n,))

    def incident_impact(self) -> pd.DataFrame:
        """Does an active incident measurably change the traffic state?"""
        return self.q("""
            SELECT CASE WHEN f.incident_count > 0 THEN 'Incident hour' ELSE 'Normal hour' END AS state,
                   COUNT(*)                         AS observations,
                   ROUND(AVG(f.congestion_index),4) AS avg_congestion,
                   ROUND(AVG(f.avg_speed_kmph),2)   AS avg_speed,
                   ROUND(AVG(f.travel_time_min),3)  AS avg_travel_time
            FROM fact_traffic f GROUP BY state
        """)


def run_all(config_path: str | None = None) -> dict[str, pd.DataFrame]:
    cfg = load_config(config_path)
    out_dir = Path(cfg["reports"]["dir"]) / "olap"
    out_dir.mkdir(parents=True, exist_ok=True)

    with TrafficCube(cfg["warehouse"]["path"]) as cube:
        results = {
            "rollup_zone_month": cube.roll_up_zone_month(),
            "rollup_city_quarter": cube.roll_up_city_quarter(),
            "drilldown_central": cube.drill_down_zone("Central"),
            "slice_heavy_rain": cube.slice_by_weather("Heavy Rain"),
            "dice_peak_core": cube.dice(["Central", "North"], ["Highway", "Arterial"], (7, 10)),
            "pivot_roadtype_hour": cube.pivot_roadtype_hour(),
            "pivot_severity_weather": cube.pivot_severity_by_weather(),
            "cube_incidents": cube.cube_incidents(),
            "top_congested_segments": cube.top_congested_segments(),
            "incident_impact": cube.incident_impact(),
        }

    for name, frame in results.items():
        frame.to_csv(out_dir / f"{name}.csv", index=frame.index.name is not None)
        print(f"[olap] {name:<24} {len(frame):>5} rows")

    print("\n[olap] Congestion by road type and hour (pivot):")
    print(results["pivot_roadtype_hour"].iloc[:, [7, 8, 12, 17, 18, 22]].to_string())
    print("\n[olap] Incident hours vs normal hours:")
    print(results["incident_impact"].to_string(index=False))
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run the OLAP cube operations.")
    ap.add_argument("--config", default=None)
    run_all(ap.parse_args().config)

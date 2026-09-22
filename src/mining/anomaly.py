"""Mining task 5 — Outlier / anomaly detection on segment-hours.

Two complementary detectors:

  * **Isolation Forest** — an unsupervised ensemble that isolates points with
    short random-partition paths; run on the multivariate traffic state.
  * **Seasonal z-score** — a statistical detector comparing each observation to
    the same segment at the same hour-of-week, which catches "this corridor is
    behaving unlike it ever does at 9 a.m. on a Tuesday".

Agreement between the two is reported: hours flagged by both are the strongest
candidates for an unreported incident or a sensor fault.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURES = ["congestion_index", "avg_speed_kmph", "occupancy_pct", "flow_per_lane", "travel_time_min"]


def run(base: pd.DataFrame, cfg: dict, contamination: float = 0.01,
        sample_size: int = 120_000) -> dict:
    seed = cfg["seed"]
    df = base.dropna(subset=FEATURES).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Seasonal z-score against the same segment at the same hour of the week.
    key = [df["segment_id"], df["timestamp"].dt.dayofweek, df["timestamp"].dt.hour]
    grp = df.groupby(key)["congestion_index"]
    df["seasonal_z"] = ((df["congestion_index"] - grp.transform("mean"))
                        / grp.transform("std").replace(0, np.nan)).fillna(0).round(3)
    df["z_anomaly"] = (df["seasonal_z"].abs() > 3).astype(int)

    # Isolation Forest on a stratified sample for tractable fitting, scored on all rows.
    fit_rows = df.sample(min(sample_size, len(df)), random_state=seed)
    scaler = StandardScaler().fit(fit_rows[FEATURES])
    iso = IsolationForest(n_estimators=200, contamination=contamination,
                          random_state=seed, n_jobs=-1).fit(scaler.transform(fit_rows[FEATURES]))
    df["iso_score"] = iso.decision_function(scaler.transform(df[FEATURES])).round(4)
    df["iso_anomaly"] = (iso.predict(scaler.transform(df[FEATURES])) == -1).astype(int)

    df["anomaly_agreement"] = df["iso_anomaly"] & df["z_anomaly"]
    flagged = df[df["anomaly_agreement"] == 1]
    unexplained = flagged[flagged["incident_count"] == 0]

    by_zone = (df.groupby("zone")["iso_anomaly"].mean() * 100).round(3).to_dict()
    top = flagged.nsmallest(20, "iso_score")[
        ["segment_id", "timestamp", "zone", "road_type", "congestion_index",
         "avg_speed_kmph", "seasonal_z", "iso_score", "incident_count"]
    ].copy()
    top["timestamp"] = top["timestamp"].astype(str)

    print(f"[anomaly] isolation-forest flags: {int(df['iso_anomaly'].sum()):,} "
          f"({df['iso_anomaly'].mean()*100:.2f}% of segment-hours)")
    print(f"[anomaly] seasonal z-score flags: {int(df['z_anomaly'].sum()):,}")
    print(f"[anomaly] both agree            : {len(flagged):,} "
          f"of which {len(unexplained):,} have no logged incident")

    return {
        "rows_scored": int(len(df)),
        "iso_anomalies": int(df["iso_anomaly"].sum()),
        "zscore_anomalies": int(df["z_anomaly"].sum()),
        "agreed_anomalies": int(len(flagged)),
        "agreed_without_logged_incident": int(len(unexplained)),
        "anomaly_rate_pct_by_zone": by_zone,
        "top_anomalies": top.to_dict(orient="records"),
        "_scored": df[["segment_id", "timestamp", "zone", "congestion_index", "seasonal_z",
                       "iso_score", "iso_anomaly", "z_anomaly", "anomaly_agreement",
                       "incident_count"]],
    }


def save(results: dict, out_dir: Path) -> Path:
    results["_scored"][results["_scored"]["iso_anomaly"] == 1].to_csv(
        out_dir / "anomalies.csv", index=False
    )
    payload = {k: v for k, v in results.items() if not k.startswith("_")}
    path = out_dir / "anomaly_results.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

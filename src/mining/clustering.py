"""Mining task 2 — Congestion hotspot discovery by clustering.

Each road segment is summarised by a behavioural profile (average and peak
congestion, peak/off-peak ratio, weekend ratio, incident rate, wet-weather
sensitivity, speed compliance) and clustered with k-Means. k is chosen by
silhouette score over the configured range; the result is a small set of named
segment archetypes the city can act on.

DBSCAN is run alongside k-Means on the geographic + congestion space to find
dense *spatial* hotspots and to label segments that belong to no cluster as
noise — a contrast between partitioning and density-based clustering.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

PROFILE_FEATURES = [
    "avg_congestion", "peak_congestion", "peak_offpeak_ratio", "weekend_ratio",
    "incident_rate_per_1k_h", "wet_penalty", "avg_speed_ratio", "pct_congested_hours",
]


def build_segment_profiles(base: pd.DataFrame) -> pd.DataFrame:
    """Reduce 8 760 hourly observations per segment to one behavioural profile."""
    g = base.groupby(["segment_id", "zone", "road_type", "lanes", "length_km",
                      "latitude", "longitude"], as_index=False)

    prof = g.agg(
        avg_congestion=("congestion_index", "mean"),
        peak_congestion=("congestion_index", lambda s: s.quantile(0.95)),
        avg_speed_ratio=("speed_ratio", "mean"),
        pct_congested_hours=("is_congested", "mean"),
        incidents=("incident_count", "sum"),
        observations=("congestion_index", "size"),
    )

    peak = base[base["is_peak_hour"] == 1].groupby("segment_id")["congestion_index"].mean()
    off = base[base["is_peak_hour"] == 0].groupby("segment_id")["congestion_index"].mean()
    wknd = base[base["is_weekend"] == 1].groupby("segment_id")["congestion_index"].mean()
    week = base[base["is_weekend"] == 0].groupby("segment_id")["congestion_index"].mean()
    wet = base[base["is_wet"] == 1].groupby("segment_id")["congestion_index"].mean()
    dry = base[base["is_wet"] == 0].groupby("segment_id")["congestion_index"].mean()

    prof["peak_offpeak_ratio"] = (prof["segment_id"].map(peak) / prof["segment_id"].map(off)).round(4)
    prof["weekend_ratio"] = (prof["segment_id"].map(wknd) / prof["segment_id"].map(week)).round(4)
    prof["wet_penalty"] = (prof["segment_id"].map(wet) - prof["segment_id"].map(dry)).round(4)
    prof["incident_rate_per_1k_h"] = (prof["incidents"] / prof["observations"] * 1000).round(3)
    prof["pct_congested_hours"] = (prof["pct_congested_hours"] * 100).round(2)
    for col in ["avg_congestion", "peak_congestion", "avg_speed_ratio"]:
        prof[col] = prof[col].round(4)
    return prof.fillna(0.0)


def choose_k(X: np.ndarray, k_range: list[int], seed: int) -> tuple[int, list[dict]]:
    """Elbow + silhouette sweep; silhouette decides."""
    diagnostics = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=25, random_state=seed).fit(X)
        diagnostics.append({
            "k": k,
            "inertia": round(float(km.inertia_), 3),
            "silhouette": round(float(silhouette_score(X, km.labels_)), 4),
            "davies_bouldin": round(float(davies_bouldin_score(X, km.labels_)), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(X, km.labels_)), 2),
        })
    best = max(diagnostics, key=lambda d: d["silhouette"])["k"]
    return best, diagnostics


def _name_clusters(summary: pd.DataFrame) -> dict[int, str]:
    """Give each cluster a human label from where it sits on the key measures."""
    names: dict[int, str] = {}
    cong_rank = summary["avg_congestion"].rank(pct=True)
    for cid, row in summary.iterrows():
        c, peaky = cong_rank.loc[cid], row["peak_offpeak_ratio"]
        if c >= 0.75 and peaky >= summary["peak_offpeak_ratio"].median():
            label = "Chronic peak-hour bottleneck"
        elif c >= 0.75:
            label = "All-day saturated corridor"
        elif c >= 0.4 and peaky >= summary["peak_offpeak_ratio"].quantile(0.6):
            label = "Commuter-driven corridor"
        elif row["incident_rate_per_1k_h"] >= summary["incident_rate_per_1k_h"].quantile(0.75):
            label = "Incident-prone but free-flowing"
        else:
            label = "Stable low-stress segment"
        names[cid] = label
    return names


def run(base: pd.DataFrame, cfg: dict) -> dict:
    seed = cfg["seed"]
    profiles = build_segment_profiles(base)

    scaler = StandardScaler()
    X = scaler.fit_transform(profiles[PROFILE_FEATURES])

    k, diagnostics = choose_k(X, cfg["mining"]["clustering_k_range"], seed)
    km = KMeans(n_clusters=k, n_init=50, random_state=seed).fit(X)
    profiles["cluster"] = km.labels_

    summary = profiles.groupby("cluster")[PROFILE_FEATURES].mean().round(4)
    summary["segments"] = profiles["cluster"].value_counts().sort_index()
    labels = _name_clusters(summary)
    summary["label"] = pd.Series(labels)
    profiles["cluster_label"] = profiles["cluster"].map(labels)

    # Density-based view on the spatial + congestion space.
    geo = StandardScaler().fit_transform(
        profiles[["latitude", "longitude", "avg_congestion"]]
    )
    db = DBSCAN(eps=0.65, min_samples=3).fit(geo)
    profiles["dbscan_cluster"] = db.labels_
    n_db = int(len(set(db.labels_) - {-1}))
    noise = int((db.labels_ == -1).sum())

    # 2-D projection for plotting.
    coords = PCA(n_components=2, random_state=seed).fit_transform(X)
    profiles["pca_1"], profiles["pca_2"] = coords[:, 0].round(4), coords[:, 1].round(4)

    hotspots = profiles.nlargest(10, "avg_congestion")[
        ["segment_id", "zone", "road_type", "avg_congestion", "peak_congestion",
         "incident_rate_per_1k_h", "cluster_label"]
    ]

    print(f"[cluster] chosen k={k} (silhouette={max(d['silhouette'] for d in diagnostics):.3f})")
    for cid, row in summary.iterrows():
        print(f"[cluster]  C{cid}: {row['label']:<32} n={int(row['segments']):<3} "
              f"avg_congestion={row['avg_congestion']:.3f}")
    print(f"[cluster] DBSCAN found {n_db} dense spatial clusters, {noise} noise segments")

    return {
        "k": int(k),
        "k_diagnostics": diagnostics,
        "silhouette": round(float(silhouette_score(X, km.labels_)), 4),
        "cluster_summary": summary.reset_index().to_dict(orient="records"),
        "cluster_labels": {int(c): n for c, n in labels.items()},
        "dbscan": {"clusters": n_db, "noise_segments": noise, "eps": 0.65, "min_samples": 3},
        "top_hotspots": hotspots.to_dict(orient="records"),
        "_profiles": profiles,
    }


def save(results: dict, out_dir: Path) -> Path:
    results["_profiles"].to_csv(out_dir / "segment_clusters.csv", index=False)
    payload = {k: v for k, v in results.items() if not k.startswith("_")}
    path = out_dir / "clustering_results.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

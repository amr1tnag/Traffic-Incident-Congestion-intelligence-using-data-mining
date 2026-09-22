"""Stage 6 — static figures for the project report (PNG, written to reports/figures)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "font.size": 9,
    "axes.grid": True, "grid.alpha": 0.3, "axes.spines.top": False, "axes.spines.right": False,
})
PALETTE = ["#2563eb", "#dc2626", "#059669", "#d97706", "#7c3aed", "#0891b2"]


def _save(fig, out_dir: Path, name: str) -> Path:
    path = out_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"[viz] {path.name}")
    return path


def congestion_profiles(base: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for i, (label, sub) in enumerate([("Weekday", base[base["is_weekend"] == 0]),
                                      ("Weekend", base[base["is_weekend"] == 1])]):
        prof = sub.groupby("hour")["congestion_index"].mean()
        axes[0].plot(prof.index, prof.values, marker="o", ms=3, color=PALETTE[i], label=label)
    axes[0].set(title="Average congestion by hour of day", xlabel="Hour", ylabel="Congestion index")
    axes[0].legend(frameon=False)

    by_type = base.groupby(["road_type", "hour"])["congestion_index"].mean().unstack(0)
    for i, col in enumerate(by_type.columns):
        axes[1].plot(by_type.index, by_type[col], color=PALETTE[i % len(PALETTE)], label=col)
    axes[1].set(title="Congestion by road class", xlabel="Hour", ylabel="Congestion index")
    axes[1].legend(frameon=False, fontsize=7)
    _save(fig, out_dir, "01_congestion_profiles")


def weekly_heatmap(base: pd.DataFrame, out_dir: Path) -> None:
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    grid = base.pivot_table(index="day_name", columns="hour",
                            values="congestion_index", aggfunc="mean").reindex(order)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(grid.values, aspect="auto", cmap="YlOrRd")
    ax.set(xticks=range(0, 24, 2), yticks=range(7), title="Congestion heatmap: day of week x hour")
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_yticklabels(order, fontsize=8)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Congestion index", shrink=0.85)
    _save(fig, out_dir, "02_weekly_heatmap")


def incident_overview(incidents: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    counts = incidents["incident_type"].value_counts()
    axes[0].barh(counts.index[::-1], counts.values[::-1], color=PALETTE[0])
    axes[0].set(title="Incidents by type", xlabel="Count")

    sev = incidents["severity_label"].value_counts().reindex(
        ["Minor", "Moderate", "Serious", "Critical"]).fillna(0)
    axes[1].bar(sev.index, sev.values, color=PALETTE[1])
    axes[1].set(title="Severity distribution", ylabel="Incidents")
    axes[1].tick_params(axis="x", labelrotation=15)

    by_hour = incidents.groupby("hour").size()
    axes[2].plot(by_hour.index, by_hour.values, marker="o", ms=3, color=PALETTE[2])
    axes[2].set(title="Incidents by hour of day", xlabel="Hour", ylabel="Incidents")
    _save(fig, out_dir, "03_incident_overview")


def weather_impact(base: pd.DataFrame, out_dir: Path) -> None:
    order = ["Clear", "Cloudy", "Fog", "Rain", "Heavy Rain"]
    agg = base.groupby("weather").agg(
        congestion=("congestion_index", "mean"), speed=("avg_speed_kmph", "mean")
    ).reindex([w for w in order if w in base["weather"].unique()])
    fig, ax = plt.subplots(figsize=(6, 3.2))
    x = np.arange(len(agg))
    ax.bar(x - 0.2, agg["congestion"], 0.4, color=PALETTE[0], label="Congestion index")
    ax2 = ax.twinx()
    ax2.bar(x + 0.2, agg["speed"], 0.4, color=PALETTE[3], label="Avg speed (km/h)")
    ax2.grid(False)
    ax.set_xticks(x, agg.index, rotation=15)
    ax.set(title="Weather impact on the network", ylabel="Congestion index")
    ax2.set_ylabel("Avg speed (km/h)")
    fig.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.9, 0.95), fontsize=7)
    _save(fig, out_dir, "04_weather_impact")


def classification_figures(results: dict, out_dir: Path) -> None:
    models = results["models"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    names = list(models)
    axes[0].bar(names, [models[m]["test_accuracy"] for m in names], color=PALETTE[0], label="Accuracy")
    axes[0].bar(names, [models[m]["macro_f1"] for m in names], width=0.45,
                color=PALETTE[1], label="Macro F1")
    axes[0].set(title="Classifier comparison", ylim=(0, 1))
    axes[0].tick_params(axis="x", labelrotation=25, labelsize=7)
    axes[0].legend(frameon=False, fontsize=7)

    cm = np.array(models[results["best_model"]]["confusion_matrix"])
    im = axes[1].imshow(cm, cmap="Blues")
    axes[1].set(title=f"Confusion matrix ({results['best_model']})",
                xlabel="Predicted severity", ylabel="Actual severity")
    axes[1].set_xticks(range(len(cm)), range(1, len(cm) + 1))
    axes[1].set_yticks(range(len(cm)), range(1, len(cm) + 1))
    axes[1].grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1].text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                         color="white" if cm[i, j] > cm.max() * 0.6 else "black")
    fig.colorbar(im, ax=axes[1], shrink=0.8)

    if results.get("feature_importance"):
        imp = pd.DataFrame(results["feature_importance"]).head(10).iloc[::-1]
        axes[2].barh(imp["feature"], imp["importance"], color=PALETTE[2])
        axes[2].set(title="Top predictors of severity (Random Forest)")
        axes[2].tick_params(axis="y", labelsize=7)
    _save(fig, out_dir, "05_classification")


def clustering_figures(results: dict, out_dir: Path) -> None:
    prof, diag = results["_profiles"], pd.DataFrame(results["k_diagnostics"])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.4))
    axes[0].plot(diag["k"], diag["inertia"], marker="o", color=PALETTE[0])
    axes[0].set(title="Elbow curve", xlabel="k", ylabel="Inertia")
    axes[1].plot(diag["k"], diag["silhouette"], marker="o", color=PALETTE[1])
    axes[1].axvline(results["k"], ls="--", color="grey", lw=1)
    axes[1].set(title="Silhouette score", xlabel="k")
    for i, (cid, grp) in enumerate(prof.groupby("cluster")):
        axes[2].scatter(grp["pca_1"], grp["pca_2"], s=28, color=PALETTE[i % len(PALETTE)],
                        label=f"C{cid}: {results['cluster_labels'][int(cid)][:22]}")
    axes[2].set(title="Segment clusters (PCA projection)", xlabel="PC1", ylabel="PC2")
    axes[2].legend(frameon=False, fontsize=6)
    _save(fig, out_dir, "06_clustering")

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for i, (cid, grp) in enumerate(prof.groupby("cluster")):
        ax.scatter(grp["longitude"], grp["latitude"], s=grp["avg_congestion"] * 180,
                   color=PALETTE[i % len(PALETTE)], alpha=0.75, label=f"C{cid}")
    ax.set(title="Spatial hotspot map (marker size = avg congestion)",
           xlabel="Longitude", ylabel="Latitude")
    ax.legend(frameon=False, fontsize=7)
    _save(fig, out_dir, "07_hotspot_map")


def association_figure(results: dict, out_dir: Path) -> None:
    rules = results["_rules"].head(12)
    if rules.empty:
        return
    labels = [f"{r.antecedent} => {r.consequent}" for r in rules.itertuples()]
    labels = [l if len(l) < 58 else l[:55] + "..." for l in labels]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(labels[::-1], rules["lift"].to_numpy()[::-1], color=PALETTE[4])
    ax.axvline(1.0, ls="--", color="grey", lw=1)
    ax.set(title="Top association rules by lift", xlabel="Lift")
    ax.tick_params(axis="y", labelsize=6.5)
    _save(fig, out_dir, "08_association_rules")


def forecast_figure(results: dict, out_dir: Path) -> None:
    fc = results["_forecast"]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(fc["timestamp"], fc["actual"], color="black", lw=1.2, label="Actual")
    ax.plot(fc["timestamp"], fc["predicted"], color=PALETTE[0], lw=1.2, ls="--",
            label=f"Predicted ({results['best_model']})")
    ax.set(title=f"Congestion forecast — final {results['horizon_hours']} hours (hold-out)",
           ylabel="Congestion index")
    ax.legend(frameon=False)
    fig.autofmt_xdate()
    _save(fig, out_dir, "09_forecast")


def anomaly_figure(results: dict, out_dir: Path) -> None:
    df = results["_scored"].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    sample = df.sample(min(40_000, len(df)), random_state=0)
    fig, ax = plt.subplots(figsize=(10, 3.4))
    normal = sample[sample["iso_anomaly"] == 0]
    flagged = sample[sample["iso_anomaly"] == 1]
    ax.scatter(normal["timestamp"], normal["congestion_index"], s=1, alpha=0.15,
               color="#94a3b8", label="Normal")
    ax.scatter(flagged["timestamp"], flagged["congestion_index"], s=6, alpha=0.8,
               color=PALETTE[1], label="Anomaly")
    ax.set(title="Detected anomalous segment-hours", ylabel="Congestion index")
    ax.legend(frameon=False, markerscale=3)
    fig.autofmt_xdate()
    _save(fig, out_dir, "10_anomalies")


def generate_all(base: pd.DataFrame, incident_features: pd.DataFrame,
                 mining: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    congestion_profiles(base, out_dir)
    weekly_heatmap(base, out_dir)
    incident_overview(incident_features, out_dir)
    weather_impact(base, out_dir)
    if "classification" in mining:
        classification_figures(mining["classification"], out_dir)
    if "clustering" in mining:
        clustering_figures(mining["clustering"], out_dir)
    if "association" in mining:
        association_figure(mining["association"], out_dir)
    if "forecasting" in mining:
        forecast_figure(mining["forecasting"], out_dir)
    if "anomaly" in mining:
        anomaly_figure(mining["anomaly"], out_dir)

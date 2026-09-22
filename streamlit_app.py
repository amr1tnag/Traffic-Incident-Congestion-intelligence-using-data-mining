"""Traffic Incident & Congestion Intelligence — presentation dashboard.

Reads the artefacts already produced by `run_pipeline.py` in `reports/` and
presents the five mining tasks (classification, clustering, association
rules, forecasting, anomaly detection) plus the OLAP findings as an
interactive dashboard.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
REPORTS = ROOT / "reports"

st.set_page_config(
    page_title="Traffic intelligence",
    page_icon=":material/traffic:",
    layout="wide",
)

CLUSTER_COLORS = ["#60A5FA", "#34D399", "#A78BFA", "#FBBF24", "#F87171", "#38BDF8", "#FB923C"]


# --------------------------------------------------------------------- data

@st.cache_data(ttl="10m")
def load_json(name: str) -> dict:
    with open(REPORTS / name, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(ttl="10m")
def load_csv(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(REPORTS / name, **kwargs)


def reports_ready() -> bool:
    return (REPORTS / "findings.md").exists() and (REPORTS / "classification_results.json").exists()


# ----------------------------------------------------------- pipeline rerun

def rerun_pipeline() -> None:
    with st.status("Running the full pipeline (generate → preprocess → etl → olap → mine → visualize → report)…",
                    expanded=True) as status:
        log_slot = st.empty()
        lines: list[str] = []
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "run_pipeline.py")],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        for line in process.stdout:  # type: ignore[union-attr]
            lines.append(line.rstrip())
            log_slot.code("\n".join(lines[-24:]), language=None)
        process.wait()
        if process.returncode == 0:
            status.update(label="Pipeline finished — reloading dashboard data", state="complete")
        else:
            status.update(label=f"Pipeline exited with code {process.returncode}", state="error")
            return
    load_json.clear()
    load_csv.clear()
    st.rerun()


# ---------------------------------------------------------------- overview

def render_overview() -> None:
    prep = load_json("preprocessing_report.json")
    pivot = load_csv("olap/pivot_roadtype_hour.csv")
    zone_month = load_csv("olap/rollup_zone_month.csv")
    top_segments = load_csv("olap/top_congested_segments.csv")
    impact = load_csv("olap/incident_impact.csv")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        with st.container(border=True):
            st.subheader("Data quality handled in preprocessing", anchor=False)
            readings = prep["readings"]
            st.table(
                {
                    "Duplicate transmissions removed": f"{readings['duplicate_rows_removed']:,}",
                    "Impossible sensor values voided": f"{readings['out_of_range_values_voided']:,}",
                    "Missing values imputed": f"{readings['missing_values_imputed']:,}",
                    "Extreme values flagged as outliers": f"{readings['outliers_flagged']:,}",
                    "Weather labels normalised": f"{readings['weather_missing_before']:,}",
                },
                border="horizontal",
            )
    with col2:
        with st.container(border=True):
            st.subheader("Incident hours vs. normal hours", anchor=False)
            inc = impact[impact["state"] == "Incident hour"].iloc[0]
            nor = impact[impact["state"] == "Normal hour"].iloc[0]
            with st.container(horizontal=True):
                st.metric("Avg congestion, incident hour", f"{inc['avg_congestion']:.2f}",
                          f"{inc['avg_congestion'] - nor['avg_congestion']:+.2f} vs normal", border=True)
                st.metric("Avg speed, incident hour", f"{inc['avg_speed']:.1f} km/h",
                          f"{inc['avg_speed'] - nor['avg_speed']:+.1f} vs normal", delta_color="inverse", border=True)
            st.caption(f"{inc['observations']:,} incident hours vs {nor['observations']:,} normal hours, one year.")

    with st.container(border=True):
        st.subheader("Congestion by road type and hour", anchor=False)
        long = pivot.melt(id_vars="road_type", var_name="hour", value_name="congestion")
        long["hour"] = long["hour"].astype(int)
        heat = (
            alt.Chart(long)
            .mark_rect()
            .encode(
                x=alt.X("hour:O", title="Hour of day"),
                y=alt.Y("road_type:N", title=None),
                color=alt.Color("congestion:Q", title="Congestion index",
                                scale=alt.Scale(scheme="blues")),
                tooltip=["road_type", "hour", alt.Tooltip("congestion:Q", format=".2f")],
            )
            .properties(height=180)
        )
        st.altair_chart(heat)

    col3, col4 = st.columns(2, gap="large")
    with col3:
        with st.container(border=True):
            st.subheader("Average congestion by zone, month over month", anchor=False)
            chart = (
                alt.Chart(zone_month)
                .mark_line(point=True)
                .encode(
                    x=alt.X("month:O", title="Month"),
                    y=alt.Y("avg_congestion:Q", title="Avg congestion"),
                    color=alt.Color("zone:N", title="Zone"),
                    tooltip=["zone", "month_name", alt.Tooltip("avg_congestion:Q", format=".3f")],
                )
                .properties(height=280)
            )
            st.altair_chart(chart)
    with col4:
        with st.container(border=True):
            st.subheader("Ten most congested segments", anchor=False)
            st.dataframe(
                top_segments,
                hide_index=True,
                column_config={
                    "segment_id": "Segment",
                    "zone": "Zone",
                    "road_type": "Road type",
                    "avg_congestion": st.column_config.NumberColumn("Avg congestion", format="%.3f"),
                    "pct_congested_hours": st.column_config.ProgressColumn(
                        "% hours congested", min_value=0, max_value=100, format="%.1f%%"),
                    "incidents": st.column_config.NumberColumn("Incidents"),
                },
            )


# ----------------------------------------------------------- classification

def render_classification() -> None:
    results = load_json("classification_results.json")
    models = results["models"]
    best = results["best_model"]

    model_df = pd.DataFrame([
        {"model": name, "macro_f1": m["macro_f1"], "test_accuracy": m["test_accuracy"]}
        for name, m in models.items()
    ]).sort_values("macro_f1", ascending=False)

    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        with st.container(border=True):
            st.subheader("Model comparison (macro-F1, handles class imbalance)", anchor=False)
            chart = (
                alt.Chart(model_df)
                .mark_bar()
                .encode(
                    x=alt.X("macro_f1:Q", title="Macro-F1", scale=alt.Scale(domain=[0, 1])),
                    y=alt.Y("model:N", sort="-x", title=None),
                    color=alt.condition(
                        alt.datum.model == best, alt.value("#34D399"), alt.value("#60A5FA")),
                    tooltip=["model", alt.Tooltip("macro_f1:Q", format=".3f"),
                            alt.Tooltip("test_accuracy:Q", format=".3f")],
                )
                .properties(height=220)
            )
            st.altair_chart(chart)
            st.caption(f"Best model: **{best}** — chosen on macro-F1, not raw accuracy, "
                      f"because severity classes are imbalanced.")
    with col2:
        with st.container(border=True):
            st.subheader("Feature importance", anchor=False)
            fi = pd.DataFrame(results["feature_importance"]).head(10).sort_values("importance")
            chart = (
                alt.Chart(fi)
                .mark_bar(color="#60A5FA")
                .encode(
                    x=alt.X("importance:Q", title=None),
                    y=alt.Y("feature:N", sort="-x", title=None),
                    tooltip=["feature", alt.Tooltip("importance:Q", format=".3f")],
                )
                .properties(height=280)
            )
            st.altair_chart(chart)

    best_model = models[best]
    col3, col4 = st.columns([2, 3], gap="large")
    with col3:
        with st.container(border=True):
            st.subheader(f"Per-class performance — {best}", anchor=False)
            rows = []
            for cls in ["1", "2", "3", "4"]:
                pc = best_model["per_class"][cls]
                rows.append({"severity": f"Class {cls}", "precision": pc["precision"],
                            "recall": pc["recall"], "f1": pc["f1-score"], "support": int(pc["support"])})
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                column_config={
                    "severity": "Severity",
                    "precision": st.column_config.NumberColumn(format="%.2f"),
                    "recall": st.column_config.NumberColumn(format="%.2f"),
                    "f1": st.column_config.NumberColumn(format="%.2f"),
                    "support": st.column_config.NumberColumn("Test rows"),
                },
            )
    with col4:
        with st.container(border=True):
            st.subheader(f"Confusion matrix — {best}", anchor=False)
            cm = best_model["confusion_matrix"]
            cm_rows = [
                {"actual": f"Class {i+1}", "predicted": f"Class {j+1}", "count": cm[i][j]}
                for i in range(len(cm)) for j in range(len(cm[i]))
            ]
            cm_df = pd.DataFrame(cm_rows)
            base = alt.Chart(cm_df).encode(
                x=alt.X("predicted:N", title="Predicted"),
                y=alt.Y("actual:N", title="Actual", sort="-y"),
            )
            heat = base.mark_rect().encode(
                color=alt.Color("count:Q", title="Count", scale=alt.Scale(scheme="blues")))
            text = base.mark_text(color="white").encode(text="count:Q")
            st.altair_chart((heat + text).properties(height=280))


# ---------------------------------------------------------------- clustering

def render_clustering() -> None:
    results = load_json("clustering_results.json")
    segments = load_csv("segment_clusters.csv")

    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        with st.container(border=True):
            st.subheader("Choosing k: silhouette score", anchor=False)
            diag = pd.DataFrame(results["k_diagnostics"])
            chosen_k = results["k"]
            line = alt.Chart(diag).mark_line(point=True, color="#60A5FA").encode(
                x=alt.X("k:O", title="k"), y=alt.Y("silhouette:Q", title="Silhouette score"),
                tooltip=["k", alt.Tooltip("silhouette:Q", format=".3f")],
            )
            chosen = alt.Chart(diag[diag["k"] == chosen_k]).mark_point(
                size=200, color="#34D399", filled=True).encode(x="k:O", y="silhouette:Q")
            st.altair_chart((line + chosen).properties(height=240))
            st.caption(f"k={chosen_k} chosen — highest silhouette score "
                      f"({results['silhouette']:.3f}). DBSCAN independently found "
                      f"{results['dbscan']['clusters']} dense spatial clusters "
                      f"and {results['dbscan']['noise_segments']} noise segments.")
    with col2:
        with st.container(border=True):
            st.subheader("Cluster profiles", anchor=False)
            for c in results["cluster_summary"]:
                with st.container(border=True):
                    st.markdown(f"**{c['label']}** · {c['segments']} segments")
                    st.caption(f"avg congestion {c['avg_congestion']:.2f} · "
                              f"peak {c['peak_congestion']:.2f} · "
                              f"incidents/1k h {c['incident_rate_per_1k_h']:.1f}")

    col3, col4 = st.columns(2, gap="large")
    with col3:
        with st.container(border=True):
            st.subheader("Segments in PCA space", anchor=False)
            chart = (
                alt.Chart(segments)
                .mark_circle(size=90)
                .encode(
                    x=alt.X("pca_1:Q", title="PC 1"), y=alt.Y("pca_2:Q", title="PC 2"),
                    color=alt.Color("cluster_label:N", title="Cluster",
                                    scale=alt.Scale(range=CLUSTER_COLORS)),
                    tooltip=["segment_id", "zone", "road_type",
                            alt.Tooltip("avg_congestion:Q", format=".3f")],
                )
                .properties(height=320)
            )
            st.altair_chart(chart)
    with col4:
        with st.container(border=True):
            st.subheader("Hotspot map", anchor=False)
            labels = sorted(segments["cluster_label"].unique())
            color_map = {label: CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i, label in enumerate(labels)}
            map_df = segments.copy()
            map_df["color"] = map_df["cluster_label"].map(color_map)
            st.map(map_df, latitude="latitude", longitude="longitude", color="color", size=80)
            legend = "  ".join(f":material/circle: {label}" for label in labels)
            st.caption(legend)

    with st.container(border=True):
        st.subheader("Top hotspots", anchor=False)
        hotspots = pd.DataFrame(results["top_hotspots"])
        st.dataframe(
            hotspots, hide_index=True,
            column_config={
                "segment_id": "Segment", "zone": "Zone", "road_type": "Road type",
                "avg_congestion": st.column_config.NumberColumn("Avg congestion", format="%.3f"),
                "peak_congestion": st.column_config.NumberColumn("Peak congestion", format="%.3f"),
                "incident_rate_per_1k_h": st.column_config.NumberColumn("Incidents / 1k h", format="%.1f"),
                "cluster_label": "Cluster",
            },
        )


# --------------------------------------------------------------- association

@st.fragment
def render_association() -> None:
    results = load_json("association_results.json")
    rules = load_csv("association_rules.csv")

    with st.container(horizontal=True):
        st.metric("Transactions mined", f"{results['n_transactions']:,}", border=True)
        st.metric("Frequent itemsets", sum(results["frequent_itemsets_by_size"].values()), border=True)
        st.metric("Rules passing thresholds", results["n_rules"], border=True)

    with st.container(border=True):
        st.subheader("Explore the rules", anchor=False)
        with st.popover("Filters", type="tertiary", icon=":material/filter_list:"):
            min_conf = st.slider("Min confidence", 0.0, 1.0, 0.55, 0.05)
            min_lift = st.slider("Min lift", 1.0, float(rules["lift"].max()), 1.1, 0.1)
        filtered = rules[(rules["confidence"] >= min_conf) & (rules["lift"] >= min_lift)] \
            .sort_values("lift", ascending=False)
        st.caption(f"{len(filtered):,} of {len(rules):,} rules match.")
        st.dataframe(
            filtered.head(200), hide_index=True,
            column_config={
                "antecedent": "If", "consequent": "Then",
                "support": st.column_config.NumberColumn(format="percent"),
                "confidence": st.column_config.NumberColumn(format="percent"),
                "lift": st.column_config.NumberColumn(format="%.2f"),
                "leverage": st.column_config.NumberColumn(format="%.3f"),
                "conviction": st.column_config.NumberColumn(format="%.2f"),
                "antecedent_size": None,
            },
        )

    with st.container(border=True):
        st.subheader("Confidence vs. lift (top 150 rules by lift)", anchor=False)
        top = rules.sort_values("lift", ascending=False).head(150)
        chart = (
            alt.Chart(top)
            .mark_circle()
            .encode(
                x=alt.X("confidence:Q", title="Confidence", scale=alt.Scale(zero=False)),
                y=alt.Y("lift:Q", title="Lift"),
                size=alt.Size("support:Q", title="Support"),
                color=alt.Color("antecedent_size:N", title="Antecedent size",
                                scale=alt.Scale(range=CLUSTER_COLORS)),
                tooltip=["antecedent", "consequent", alt.Tooltip("support:Q", format=".3f"),
                        alt.Tooltip("confidence:Q", format=".3f"), alt.Tooltip("lift:Q", format=".2f")],
            )
            .properties(height=320)
        )
        st.altair_chart(chart)


# --------------------------------------------------------------- forecasting

def render_forecasting() -> None:
    results = load_json("forecasting_results.json")
    forecast = load_csv("congestion_forecast.csv", parse_dates=["timestamp"])
    best = results["best_model"]

    with st.container(horizontal=True):
        st.metric("Forecast horizon", f"{results['horizon_hours']} h", border=True)
        st.metric("Best model", best, border=True)
        st.metric("Improvement over persistence baseline",
                  f"{results['improvement_over_persistence_pct']:.1f}%", border=True)

    with st.container(border=True):
        st.subheader("Actual vs. predicted congestion — final week (hold-out)", anchor=False)
        long = forecast.melt(id_vars="timestamp", value_vars=["actual", "predicted", "persistence"],
                             var_name="series", value_name="congestion")
        chart = (
            alt.Chart(long)
            .mark_line()
            .encode(
                x=alt.X("timestamp:T", title=None),
                y=alt.Y("congestion:Q", title="Congestion index"),
                color=alt.Color("series:N", title=None,
                                scale=alt.Scale(domain=["actual", "predicted", "persistence"],
                                                range=["#F1F5F9", "#34D399", "#F87171"])),
                strokeDash=alt.condition(alt.datum.series == "actual", alt.value([0, 0]), alt.value([4, 2])),
                tooltip=["timestamp:T", "series:N", alt.Tooltip("congestion:Q", format=".3f")],
            )
            .properties(height=300)
        )
        st.altair_chart(chart)
        st.caption("Hold-out is the chronologically final week — no random split, no future leakage.")

    col1, col2 = st.columns([3, 2], gap="large")
    with col1:
        with st.container(border=True):
            st.subheader("Model error (MAE, lower is better)", anchor=False)
            metrics_df = pd.DataFrame([{"model": name, **vals} for name, vals in results["metrics"].items()])
            chart = (
                alt.Chart(metrics_df)
                .mark_bar()
                .encode(
                    x=alt.X("mae:Q", title="MAE"),
                    y=alt.Y("model:N", sort="x", title=None),
                    color=alt.condition(alt.datum.model == best, alt.value("#34D399"), alt.value("#60A5FA")),
                    tooltip=["model", alt.Tooltip("mae:Q", format=".4f"), alt.Tooltip("r2:Q", format=".3f")],
                )
                .properties(height=220)
            )
            st.altair_chart(chart)
    with col2:
        with st.container(border=True):
            st.subheader("All metrics", anchor=False)
            st.dataframe(
                metrics_df.sort_values("mae"), hide_index=True,
                column_config={
                    "model": "Model",
                    "mae": st.column_config.NumberColumn("MAE", format="%.4f"),
                    "rmse": st.column_config.NumberColumn("RMSE", format="%.4f"),
                    "mape_pct": st.column_config.NumberColumn("MAPE %", format="%.1f"),
                    "r2": st.column_config.NumberColumn("R²", format="%.3f"),
                },
            )


# ----------------------------------------------------------------- anomalies

def render_anomalies() -> None:
    results = load_json("anomaly_results.json")
    anomalies = load_csv("anomalies.csv", parse_dates=["timestamp"])

    with st.container(horizontal=True):
        st.metric("Segment-hours scored", f"{results['rows_scored']:,}", border=True)
        st.metric("Isolation forest flags", f"{results['iso_anomalies']:,}", border=True)
        st.metric("Seasonal z-score flags", f"{results['zscore_anomalies']:,}", border=True)
        st.metric("Both agree", f"{results['agreed_anomalies']:,}", border=True)
        st.metric("Agree, no logged incident", f"{results['agreed_without_logged_incident']:,}",
                  border=True, help="Both detectors flag the hour, but no incident was recorded — "
                                    "the operationally interesting case: a possible unreported event.")

    col1, col2 = st.columns([2, 3], gap="large")
    with col1:
        with st.container(border=True):
            st.subheader("Anomaly rate by zone", anchor=False)
            rate_df = pd.DataFrame(
                list(results["anomaly_rate_pct_by_zone"].items()), columns=["zone", "rate_pct"]
            ).sort_values("rate_pct", ascending=False)
            chart = (
                alt.Chart(rate_df)
                .mark_bar(color="#60A5FA")
                .encode(x=alt.X("rate_pct:Q", title="% segment-hours flagged"),
                        y=alt.Y("zone:N", sort="-x", title=None),
                        tooltip=["zone", alt.Tooltip("rate_pct:Q", format=".2f")])
                .properties(height=220)
            )
            st.altair_chart(chart)
    with col2:
        with st.container(border=True):
            st.subheader("Flagged hours", anchor=False)
            only_unreported = st.checkbox(
                "Only hours both detectors flag with no logged incident", value=True)
            agreed = anomalies[anomalies["anomaly_agreement"] == 1]
            shown = agreed[agreed["incident_count"] == 0] if only_unreported else agreed
            shown = shown.sort_values("seasonal_z", ascending=False).head(200)
            st.dataframe(
                shown, hide_index=True,
                column_config={
                    "segment_id": "Segment", "timestamp": st.column_config.DatetimeColumn("When", format="MMM D, HH:mm"),
                    "zone": "Zone",
                    "congestion_index": st.column_config.NumberColumn("Congestion", format="%.2f"),
                    "seasonal_z": st.column_config.NumberColumn("Seasonal z", format="%.2f"),
                    "iso_score": st.column_config.NumberColumn("Isolation score", format="%.3f"),
                    "iso_anomaly": None, "z_anomaly": None, "anomaly_agreement": None,
                    "incident_count": st.column_config.NumberColumn("Logged incidents"),
                },
            )


# --------------------------------------------------------------------- main

st.title("Traffic incident & congestion intelligence", icon=":material/traffic:")
st.caption("Raw feeds → cleaning → star-schema warehouse → OLAP → five mining techniques → this dashboard. "
          "All figures below are read live from `reports/`.")

with st.sidebar:
    st.subheader("About", anchor=False)
    st.caption("A year of simulated traffic for a 60-segment network: "
              "525,600 segment-hours and 12,218 incidents.")
    st.caption("Data warehousing & data mining capstone.")
    st.space("small")
    if reports_ready():
        mtime = (REPORTS / "findings.md").stat().st_mtime
        st.caption(f"Report last generated: {pd.Timestamp(mtime, unit='s'):%Y-%m-%d %H:%M}")
    with st.expander("Rerun the pipeline", icon=":material/refresh:"):
        st.caption("Regenerates the simulated year, re-cleans it, reloads the warehouse, "
                  "reruns all five mining tasks and rebuilds this report. Takes about a minute.")
        if st.button("Run full pipeline", icon=":material/play_arrow:", type="primary"):
            rerun_pipeline()

if not reports_ready():
    st.warning("No reports found yet. Run the pipeline first: `python run_pipeline.py`, "
              "or use **Rerun the pipeline** in the sidebar.", icon=":material/warning:")
    st.stop()

tabs = st.tabs([
    ":material/dashboard: Overview",
    ":material/rule: Classification",
    ":material/scatter_plot: Clustering",
    ":material/link: Association rules",
    ":material/trending_up: Forecasting",
    ":material/report: Anomalies",
])

with tabs[0]:
    render_overview()
with tabs[1]:
    render_classification()
with tabs[2]:
    render_clustering()
with tabs[3]:
    render_association()
with tabs[4]:
    render_forecasting()
with tabs[5]:
    render_anomalies()

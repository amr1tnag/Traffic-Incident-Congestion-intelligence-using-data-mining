"""Stage 7 — assemble every stage's output into reports/findings.md."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd


def _table(df: pd.DataFrame, max_rows: int = 12) -> str:
    return df.head(max_rows).to_markdown(index=False)


def build(cfg: dict, base: pd.DataFrame, incidents: pd.DataFrame,
          olap: dict, mining: dict) -> Path:
    r = Path(cfg["reports"]["dir"])
    pre = json.loads((r / "preprocessing_report.json").read_text(encoding="utf-8"))
    L = []
    A = L.append

    A("# Traffic Incident & Congestion Intelligence — Findings\n")
    A(f"_Generated {datetime.now():%Y-%m-%d %H:%M} from the automated pipeline._\n")

    A("## 1. Dataset and warehouse\n")
    A(f"- Observation window: **{base['date'].min()} to {base['date'].max()}** "
      f"({cfg['data']['n_days']} days, hourly)")
    A(f"- Road network: **{base['segment_id'].nunique()} segments** across "
      f"{base['zone'].nunique()} zones and {base['road_type'].nunique()} road classes")
    A(f"- Fact rows: **{len(base):,}** segment-hours and **{len(incidents):,}** incidents")
    A(f"- Warehouse: star schema at `{Path(cfg['warehouse']['path']).name}` "
      "(4 dimensions, 2 fact tables, 3 materialised aggregates)\n")

    A("## 2. Data quality handled in preprocessing\n")
    A(f"| Issue | Records |\n|---|---|")
    A(f"| Duplicate transmissions removed | {pre['readings']['duplicate_rows_removed']:,} |")
    A(f"| Impossible sensor values voided | {pre['readings']['out_of_range_values_voided']:,} |")
    A(f"| Missing values imputed | {pre['readings']['missing_values_imputed'] + pre['incidents']['missing_values_imputed']:,} |")
    A(f"| Extreme values flagged as outliers | {pre['readings']['outliers_flagged']:,} |")
    A(f"| Weather labels unified | {pre['readings']['weather_missing_before']:,} missing + case variants |\n")

    A("## 3. OLAP analysis\n")
    A("**Incident hours versus normal hours**\n")
    A(_table(olap["incident_impact"]) + "\n")
    A("**Ten most congested segments (roll-up over the year)**\n")
    A(_table(olap["top_congested_segments"], 10) + "\n")
    A("**Congestion by road class and hour (pivot, selected hours)**\n")
    piv = olap["pivot_roadtype_hour"]
    A(piv[[c for c in piv.columns if c in (3, 8, 12, 18, 22)]].to_markdown() + "\n")
    A("**Adverse-weather slice (Heavy Rain)**\n")
    A(_table(olap["slice_heavy_rain"]) + "\n")

    if "classification" in mining:
        c = mining["classification"]
        A("## 4. Incident severity classification\n")
        rows = [{"model": m, **{k: v for k, v in r.items()
                                if k in ("cv_accuracy_mean", "test_accuracy", "macro_precision",
                                         "macro_recall", "macro_f1")}}
                for m, r in c["models"].items()]
        A(_table(pd.DataFrame(rows)) + "\n")
        A(f"Best model: **{c['best_model']}** "
          f"(macro-F1 {c['models'][c['best_model']]['macro_f1']:.3f}, "
          f"accuracy {c['models'][c['best_model']]['test_accuracy']:.3f}).\n")
        if c.get("feature_importance"):
            A("**Strongest predictors**\n")
            A(_table(pd.DataFrame(c["feature_importance"]), 10) + "\n")
        if c.get("decision_rules"):
            A("**Decision tree (top levels)**\n")
            A("```\n" + "\n".join(c["decision_rules"].splitlines()[:30]) + "\n```\n")

    if "clustering" in mining:
        cl = mining["clustering"]
        A("## 5. Congestion hotspot clustering\n")
        A(f"k-Means selected **k = {cl['k']}** by silhouette ({cl['silhouette']:.3f}); "
          f"DBSCAN found {cl['dbscan']['clusters']} dense spatial clusters "
          f"and {cl['dbscan']['noise_segments']} noise segments.\n")
        summ = pd.DataFrame(cl["cluster_summary"])
        A(_table(summ[["cluster", "label", "segments", "avg_congestion", "peak_congestion",
                       "peak_offpeak_ratio", "incident_rate_per_1k_h"]]) + "\n")
        A("**Top hotspot segments**\n")
        A(_table(pd.DataFrame(cl["top_hotspots"]), 10) + "\n")

    if "association" in mining:
        a = mining["association"]
        A("## 6. Association rules (Apriori)\n")
        A(f"{a['n_transactions']:,} incident transactions; frequent itemsets "
          f"{a['frequent_itemsets_by_size']}; **{a['n_rules']} rules** at "
          f"support ≥ {a['parameters']['min_support']}, confidence ≥ "
          f"{a['parameters']['min_confidence']}, lift ≥ {a['parameters']['min_lift']}.\n")
        A("**Top rules by lift**\n")
        A(_table(pd.DataFrame(a["top_rules"])[
            ["antecedent", "consequent", "support", "confidence", "lift", "leverage"]], 12) + "\n")
        if a["severity_rules"]:
            A("**Rules that conclude a severity level**\n")
            A(_table(pd.DataFrame(a["severity_rules"])[
                ["antecedent", "consequent", "support", "confidence", "lift"]], 10) + "\n")

    if "forecasting" in mining:
        f = mining["forecasting"]
        A("## 7. Congestion forecasting\n")
        A(f"Chronological hold-out of the final {f['horizon_hours']} hours "
          f"({f['train_rows']:,} training hours).\n")
        A(_table(pd.DataFrame(f["metrics"]).T.reset_index().rename(columns={"index": "model"}), 8) + "\n")
        A(f"Best model **{f['best_model']}** cuts MAE by "
          f"**{f['improvement_over_persistence_pct']:.1f}%** against the persistence baseline.\n")

    if "anomaly" in mining:
        an = mining["anomaly"]
        A("## 8. Anomaly detection\n")
        A(f"- Isolation Forest flagged **{an['iso_anomalies']:,}** of {an['rows_scored']:,} segment-hours")
        A(f"- Seasonal z-score flagged **{an['zscore_anomalies']:,}**")
        A(f"- Both agreed on **{an['agreed_anomalies']:,}**, of which "
          f"**{an['agreed_without_logged_incident']:,}** have no logged incident — "
          "candidates for unreported events or sensor faults\n")
        A(_table(pd.DataFrame(an["top_anomalies"]), 10) + "\n")

    A("## 9. Figures\n")
    for png in sorted(Path(cfg["reports"]["figures_dir"]).glob("*.png")):
        A(f"- `figures/{png.name}`")
    A("")

    path = r / "findings.md"
    path.write_text("\n".join(L), encoding="utf-8")
    print(f"[report] {path}")
    return path

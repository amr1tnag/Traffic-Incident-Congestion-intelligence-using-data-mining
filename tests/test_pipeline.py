"""Unit tests for the core pipeline logic (run with: pytest -q)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from src.mining.association import apriori, build_transactions, generate_rules
from src.mining.clustering import build_segment_profiles
from src.mining.forecasting import make_supervised
from src.preprocess import (VALID_RANGES, _normalise_weather, add_time_features,
                            clean_readings, discretise_congestion)


# ------------------------------------------------------------- preprocessing
def _readings(n: int = 240) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "segment_id": ["SEG001"] * n,
        "timestamp": ts,
        "vehicle_count": rng.integers(100, 900, n).astype(float),
        "avg_speed_kmph": rng.uniform(20, 60, n),
        "occupancy_pct": rng.uniform(5, 80, n),
        "travel_time_min": rng.uniform(1, 9, n),
        "congestion_index": rng.uniform(0.05, 0.95, n),
        "weather": ["Clear"] * n,
        "temperature_c": rng.uniform(15, 35, n),
        "precipitation_mm": np.zeros(n),
        "incident_active": np.zeros(n, dtype=int),
    })


def test_normalise_weather_folds_case_and_spacing():
    s = pd.Series(["HEAVY RAIN", "heavy rain", "  Clear ", None])
    out = _normalise_weather(s)
    assert out.tolist()[:3] == ["Heavy Rain", "Heavy Rain", "Clear"]
    assert pd.isna(out.iloc[3])


def test_clean_readings_removes_duplicates():
    df = _readings()
    df = pd.concat([df, df.head(10)], ignore_index=True)
    cleaned, log = clean_readings(df, outlier_z=4.0)
    assert log["duplicate_rows_removed"] == 10
    assert cleaned.duplicated(subset=["segment_id", "timestamp"]).sum() == 0


def test_clean_readings_voids_impossible_values_then_imputes():
    df = _readings()
    df.loc[5, "avg_speed_kmph"] = -1.0          # impossible
    df.loc[6, "occupancy_pct"] = 150.0          # impossible
    df.loc[7, "vehicle_count"] = np.nan         # missing
    cleaned, log = clean_readings(df, outlier_z=4.0)
    assert log["out_of_range_values_voided"] == 2
    assert cleaned[list(VALID_RANGES)].isna().sum().sum() == 0
    lo, hi = VALID_RANGES["avg_speed_kmph"]
    assert cleaned["avg_speed_kmph"].between(lo, hi).all()


def test_time_features_are_consistent():
    out = add_time_features(_readings())
    assert out["hour"].between(0, 23).all()
    assert set(out["is_weekend"].unique()) <= {0, 1}
    # cyclical encodings must lie on the unit circle
    assert np.allclose(out["hour_sin"] ** 2 + out["hour_cos"] ** 2, 1.0)


def test_discretisation_respects_bin_edges():
    df = pd.DataFrame({"congestion_index": [0.0, 0.2, 0.5, 0.7, 0.99]})
    bins = [0.0, 0.35, 0.60, 0.80, 1.01]
    labels = ["Free flow", "Light", "Moderate", "Severe"]
    out = discretise_congestion(df, bins, labels)
    assert out["congestion_level"].tolist() == [
        "Free flow", "Free flow", "Light", "Moderate", "Severe"]
    assert out["is_congested"].tolist() == [0, 0, 0, 1, 1]


# ------------------------------------------------------------------ Apriori
def test_apriori_support_matches_brute_force():
    transactions = [
        {"A", "B", "C"}, {"A", "B"}, {"A", "C"}, {"B", "C"}, {"A", "B", "C"}, {"A"},
    ]
    frequent = apriori(transactions, min_support=0.5, max_len=3)
    assert frequent[frozenset(["A"])] == pytest.approx(5 / 6)
    assert frequent[frozenset(["A", "B"])] == pytest.approx(3 / 6)
    # {A,B,C} has support 2/6 < 0.5 and must be pruned
    assert frozenset(["A", "B", "C"]) not in frequent


def test_apriori_downward_closure_holds():
    rng = np.random.default_rng(1)
    items = list("ABCDEF")
    transactions = [set(rng.choice(items, rng.integers(2, 5), replace=False)) for _ in range(300)]
    frequent = apriori(transactions, min_support=0.1, max_len=3)
    for itemset in frequent:
        for item in itemset:
            assert frozenset([item]) in frequent, "a subset of a frequent itemset must be frequent"


def test_rule_metrics_are_correct():
    transactions = [{"A", "B"}] * 6 + [{"A"}] * 2 + [{"B"}] * 2
    frequent = apriori(transactions, min_support=0.3, max_len=2)
    # min_lift is relaxed here so the metric arithmetic itself can be asserted
    rules = generate_rules(frequent, min_confidence=0.5, min_lift=0.0)
    rule = rules[(rules["antecedent"] == "A") & (rules["consequent"] == "B")].iloc[0]
    assert rule["support"] == pytest.approx(0.6)
    assert rule["confidence"] == pytest.approx(0.75)     # 0.6 / 0.8
    assert rule["lift"] == pytest.approx(0.75 / 0.8)


def test_build_transactions_produces_prefixed_items():
    df = pd.DataFrame({
        "incident_type": ["Collision"], "weather": ["Rain"], "road_type": ["Highway"],
        "zone": ["North"], "time_of_day": ["Evening"], "is_peak_hour": [1], "is_weekend": [0],
        "congestion_level": ["Severe"], "lanes_blocked": [2], "vehicles_involved": [3],
        "duration_band": ["45-90m"], "response_band": ["Slow"], "severity_label": ["Serious"],
    })
    tx = build_transactions(df)[0]
    assert "TYPE=Collision" in tx and "SEVERITY=Serious" in tx and "LANES_BLOCKED=2+" in tx
    assert all("=" in item for item in tx)


# ------------------------------------------------------------ other mining
def test_segment_profiles_one_row_per_segment():
    base = add_time_features(_readings())
    base = discretise_congestion(base, [0.0, 0.35, 0.60, 0.80, 1.01],
                                 ["Free flow", "Light", "Moderate", "Severe"])
    base = base.assign(zone="North", road_type="Arterial", lanes=2, length_km=1.0,
                       latitude=12.9, longitude=77.6, speed_ratio=0.6, incident_count=0,
                       is_wet=0)
    profiles = build_segment_profiles(base)
    assert len(profiles) == base["segment_id"].nunique()
    assert profiles["avg_congestion"].between(0, 1).all()


def test_supervised_frame_has_no_future_leakage():
    series = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=400, freq="h"),
        "congestion_index": np.linspace(0.1, 0.9, 400),
        "avg_speed": np.linspace(50, 20, 400),
        "incidents": np.zeros(400),
        "is_wet": np.zeros(400),
    })
    sup = make_supervised(series, [1, 2, 24, 168])
    assert not sup.isna().any().any()
    row = sup.iloc[10]
    original = series.set_index("timestamp")["congestion_index"]
    assert row["lag_1"] == pytest.approx(original.loc[row["timestamp"] - pd.Timedelta(hours=1)])


# --------------------------------------------------------------- Power BI export
def test_dim_date_is_contiguous_and_unique():
    """Power BI can only mark a date table whose dates are unique and gap-free."""
    from src.export_bi import build_dim_date

    hourly = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=72, freq="h").strftime("%Y-%m-%d")
    })
    # Punch a hole: the source skips a day, dim_date must still be contiguous.
    hourly = hourly[hourly["date"] != "2024-01-02"]

    dim = build_dim_date(hourly)
    dates = pd.to_datetime(dim["date"])

    assert dim["date_key"].is_unique
    assert dates.is_unique
    assert (dates.diff().dropna() == pd.Timedelta(days=1)).all()
    assert dim.loc[dim["date"] == "2024-01-06", "day_name"].eq("Saturday").all()
    assert dim.loc[dim["date"] == "2024-01-06", "is_weekend"].eq(1).all()


def test_model_metrics_export_is_tall(tmp_path):
    """Every mining task flattens into (task, model, metric, value) rows."""
    import json

    from src.export_bi import export_model_metrics

    (tmp_path / "classification_results.json").write_text(json.dumps(
        {"models": {"decision_tree": {"test_accuracy": 0.71, "macro_f1": 0.73}}}))
    (tmp_path / "forecasting_results.json").write_text(json.dumps(
        {"metrics": {"random_forest": {"mae": 0.02, "r2": 0.97}}}))
    (tmp_path / "association_results.json").write_text(json.dumps({"n_rules": 406}))

    rows = export_model_metrics(tmp_path, tmp_path)
    frame = pd.read_csv(tmp_path / "mining_model_metrics.csv", encoding="utf-8-sig")

    assert rows == len(frame) == 5
    assert list(frame.columns) == ["task", "model", "metric", "value"]
    assert set(frame["task"]) == {"Classification", "Forecasting", "Association rules"}
    accuracy = frame.query("model == 'decision_tree' and metric == 'test_accuracy'")
    assert accuracy["value"].iloc[0] == pytest.approx(0.71)


def test_export_refuses_a_missing_warehouse(tmp_path):
    from src.export_bi import export_warehouse

    with pytest.raises(FileNotFoundError, match="run_pipeline"):
        export_warehouse(tmp_path / "absent.db", tmp_path)


# ------------------------------------------------------------- Power BI project
def test_every_measure_references_a_real_column():
    """A typo in a DAX measure only surfaces on refresh — catch it here instead."""
    import re
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "powerbi"))
    import build_pbip

    columns = _exported_columns()
    defined = {name for name, _, _, _ in build_pbip.MEASURES}

    for name, dax, _, _ in build_pbip.MEASURES:
        for table, column in re.findall(r"(\w+)\[([^\]]+)\]", dax):
            assert table in columns, f"measure {name!r} reads unknown table {table!r}"
            assert column in columns[table], \
                f"measure {name!r} reads {table!r} column {column!r}, which does not exist"
        # A bare [Name] reference is a call to another measure.
        for referenced in re.findall(r"(?<![\w\]])\[([^\]]+)\]", dax):
            assert referenced in defined, \
                f"measure {name!r} calls undefined measure [{referenced}]"


def _exported_columns() -> dict[str, set[str]]:
    """table -> column names, read from the folder `src.export_bi` writes."""
    data = Path(__file__).resolve().parents[1] / "powerbi" / "data"
    csvs = sorted(data.glob("*.csv")) if data.exists() else []
    if not csvs:
        pytest.skip("run `python -m src.export_bi` to enable the Power BI checks")
    return {path.stem: set(pd.read_csv(path, nrows=1, encoding="utf-8-sig").columns)
            for path in csvs}


def test_report_visuals_only_bind_to_modelled_fields():
    """Guards the report against drifting away from the semantic model."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "powerbi"))
    import build_pbip

    measures = {name for name, _, _, _ in build_pbip.MEASURES}
    columns = _exported_columns()

    seen = set()
    for page_id, _, visuals in build_pbip.build_pages():
        for spec in visuals:
            assert (page_id, spec["slug"]) not in seen, "duplicate visual id"
            seen.add((page_id, spec["slug"]))
            for role in spec["visual"].get("query", {}).get("queryState", {}).values():
                for proj in role["projections"]:
                    field = proj["field"]
                    if "Measure" in field:
                        assert field["Measure"]["Property"] in measures
                    else:
                        entity = field["Column"]["Expression"]["SourceRef"]["Entity"]
                        prop = field["Column"]["Property"]
                        assert entity in columns, f"unknown table {entity}"
                        assert prop in columns[entity], f"{entity} has no column {prop}"

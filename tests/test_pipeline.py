"""Unit tests for the core pipeline logic (run with: pytest -q)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

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

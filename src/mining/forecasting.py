"""Mining task 4 — Short-horizon congestion forecasting.

The city-wide hourly congestion series is modelled with lag/rolling features
(t-1, t-2, t-3, t-24, t-168 plus rolling means and calendar terms) and a
gradient-boosted regressor. Evaluation is a strictly chronological hold-out —
the last ``horizon_hours`` of the year — so no future information leaks into
training, which a random split would silently allow on time-series data.

Two naive baselines are reported for honesty:
  * persistence   — "next hour equals this hour"
  * seasonal naive— "next hour equals the same hour last week"
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def build_series(base: pd.DataFrame, segment_id: str | None = None) -> pd.DataFrame:
    """City-wide (or single-segment) hourly congestion series."""
    df = base if segment_id is None else base[base["segment_id"] == segment_id]
    series = df.groupby("timestamp").agg(
        congestion_index=("congestion_index", "mean"),
        avg_speed=("avg_speed_kmph", "mean"),
        incidents=("incident_count", "sum"),
        is_wet=("is_wet", "mean"),
    ).reset_index()
    series["timestamp"] = pd.to_datetime(series["timestamp"])
    return series.sort_values("timestamp").reset_index(drop=True)


def make_supervised(series: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Turn the series into a supervised table: y(t) from features known at t-1."""
    df = series.copy()
    for lag in lags:
        df[f"lag_{lag}"] = df["congestion_index"].shift(lag)
    df["roll_mean_3"] = df["congestion_index"].shift(1).rolling(3).mean()
    df["roll_mean_24"] = df["congestion_index"].shift(1).rolling(24).mean()
    df["roll_std_24"] = df["congestion_index"].shift(1).rolling(24).std()
    df["incidents_lag_1"] = df["incidents"].shift(1)
    df["wet_lag_1"] = df["is_wet"].shift(1)

    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = ts.dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df.dropna().reset_index(drop=True)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1e-6, None))) * 100)
    return {"mae": round(float(mae), 5), "rmse": round(rmse, 5),
            "mape_pct": round(mape, 3), "r2": round(float(r2_score(y_true, y_pred)), 4)}


def run(base: pd.DataFrame, cfg: dict) -> dict:
    seed, fc = cfg["seed"], cfg["mining"]["forecast"]
    series = build_series(base)
    sup = make_supervised(series, fc["lags"])

    feature_cols = [c for c in sup.columns
                    if c not in {"timestamp", "congestion_index", "avg_speed", "incidents", "is_wet"}]
    horizon = fc["horizon_hours"]
    train, test = sup.iloc[:-horizon], sup.iloc[-horizon:]
    X_tr, y_tr = train[feature_cols], train["congestion_index"]
    X_te, y_te = test[feature_cols], test["congestion_index"]

    models = {
        "ridge": Ridge(alpha=1.0, random_state=seed),
        "random_forest": RandomForestRegressor(n_estimators=250, max_depth=16,
                                               min_samples_leaf=3, n_jobs=-1, random_state=seed),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                                       max_depth=3, random_state=seed),
    }
    results: dict[str, dict] = {
        "persistence_baseline": _metrics(y_te.to_numpy(), test["lag_1"].to_numpy()),
        "seasonal_naive_baseline": _metrics(y_te.to_numpy(), test["lag_168"].to_numpy()),
    }
    preds: dict[str, np.ndarray] = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        results[name] = _metrics(y_te.to_numpy(), pred)
        preds[name] = pred
        print(f"[forecast] {name:<20} MAE={results[name]['mae']:.4f} "
              f"RMSE={results[name]['rmse']:.4f} R2={results[name]['r2']:.3f}")
    for name in ("persistence_baseline", "seasonal_naive_baseline"):
        print(f"[forecast] {name:<20} MAE={results[name]['mae']:.4f} R2={results[name]['r2']:.3f}")

    best = min((m for m in models), key=lambda k: results[k]["mae"])
    improvement = (1 - results[best]["mae"] / results["persistence_baseline"]["mae"]) * 100
    print(f"[forecast] best: {best} ({improvement:.1f}% lower MAE than persistence)")

    forecast_df = pd.DataFrame({
        "timestamp": test["timestamp"].to_numpy(),
        "actual": y_te.to_numpy().round(4),
        "predicted": preds[best].round(4),
        "persistence": test["lag_1"].to_numpy().round(4),
    })
    return {
        "horizon_hours": horizon,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "features": feature_cols,
        "metrics": results,
        "best_model": best,
        "improvement_over_persistence_pct": round(float(improvement), 2),
        "_forecast": forecast_df,
        "_series": series,
    }


def save(results: dict, out_dir: Path) -> Path:
    results["_forecast"].to_csv(out_dir / "congestion_forecast.csv", index=False)
    payload = {k: v for k, v in results.items() if not k.startswith("_")}
    path = out_dir / "forecasting_results.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

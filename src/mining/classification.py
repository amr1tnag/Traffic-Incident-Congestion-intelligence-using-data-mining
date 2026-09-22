"""Mining task 1 — Incident severity classification.

Business question: *given what the control room knows in the first minutes of a
report (incident type, vehicles involved, lanes blocked, weather, road class and
the live traffic state), how severe is this incident going to be?*

Severity is the 4-class police scale (1 Minor .. 4 Critical). Four classifiers
from the syllabus are compared on identical splits:

  * Decision Tree (CART, Gini)  — interpretable, yields the rule set below.
  * Random Forest               — bagged ensemble, usually the accuracy ceiling.
  * Gaussian Naive Bayes        — the probabilistic baseline.
  * Logistic Regression         — the linear discriminative baseline.

Stratified train/test split plus 5-fold cross-validation on the training half;
accuracy, macro precision/recall/F1 and the confusion matrix are reported.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

NUMERIC_FEATURES = [
    "vehicles_involved", "lanes_blocked", "blocked_lane_ratio", "lanes", "length_km",
    "speed_limit_kmph", "congestion_index", "avg_speed_kmph", "speed_ratio",
    "occupancy_pct", "flow_per_lane", "hour", "day_of_week", "is_weekend", "is_peak_hour",
]
CATEGORICAL_FEATURES = ["incident_type", "weather", "road_type", "zone", "time_of_day"]
TARGET = "severity"


def _build_models(seed: int) -> dict[str, object]:
    return {
        "decision_tree": DecisionTreeClassifier(
            criterion="gini", max_depth=8, min_samples_leaf=25,
            class_weight="balanced", random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=seed,
        ),
        "naive_bayes": GaussianNB(),
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=seed,
        ),
    }


def _preprocessor(dense: bool) -> ColumnTransformer:
    """Scale numerics, one-hot the categoricals. Naive Bayes needs a dense matrix."""
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense), CATEGORICAL_FEATURES),
    ])


def run(df: pd.DataFrame, cfg: dict) -> dict:
    seed = cfg["seed"]
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["mining"]["test_size"], stratify=y, random_state=seed
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    for name, model in _build_models(seed).items():
        if name not in cfg["mining"]["classification_models"]:
            continue
        pipe = Pipeline([("prep", _preprocessor(dense=(name == "naive_bayes"))), ("model", model)])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        results[name] = {
            "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
            "cv_accuracy_std": round(float(cv_scores.std()), 4),
            "test_accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "macro_precision": round(float(precision_score(y_test, pred, average="macro", zero_division=0)), 4),
            "macro_recall": round(float(recall_score(y_test, pred, average="macro", zero_division=0)), 4),
            "macro_f1": round(float(f1_score(y_test, pred, average="macro", zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "per_class": classification_report(y_test, pred, output_dict=True, zero_division=0),
        }
        fitted[name] = pipe
        print(f"[classify] {name:<20} cv={cv_scores.mean():.3f}±{cv_scores.std():.3f} "
              f"test={results[name]['test_accuracy']:.3f} macroF1={results[name]['macro_f1']:.3f}")

    best = max(results, key=lambda k: results[k]["macro_f1"])
    out: dict = {"models": results, "best_model": best,
                 "class_distribution": y.value_counts().sort_index().to_dict(),
                 "train_rows": int(len(X_train)), "test_rows": int(len(X_test))}

    if "random_forest" in fitted:
        out["feature_importance"] = _feature_importance(fitted["random_forest"])
    if "decision_tree" in fitted:
        out["decision_rules"] = _tree_rules(fitted["decision_tree"])
    out["_fitted"] = fitted
    print(f"[classify] best model by macro-F1: {best}")
    return out


def _feature_importance(pipe: Pipeline, top_n: int = 15) -> list[dict]:
    names = pipe.named_steps["prep"].get_feature_names_out()
    imp = pipe.named_steps["model"].feature_importances_
    order = np.argsort(imp)[::-1][:top_n]
    return [{"feature": str(names[i]).split("__", 1)[-1], "importance": round(float(imp[i]), 4)}
            for i in order]


def _tree_rules(pipe: Pipeline, max_depth: int = 4) -> str:
    names = [str(n).split("__", 1)[-1] for n in pipe.named_steps["prep"].get_feature_names_out()]
    return export_text(pipe.named_steps["model"], feature_names=names, max_depth=max_depth)


def save(results: dict, out_dir: Path) -> Path:
    payload = {k: v for k, v in results.items() if not k.startswith("_")}
    path = out_dir / "classification_results.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

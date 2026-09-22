"""Mining task 3 — Association rule mining over incident transactions.

Apriori is implemented from first principles (candidate generation, the
downward-closure pruning step, support counting) rather than pulled from a
library, because the algorithm itself is examinable material.

Each incident becomes a transaction of categorical items, e.g.

    {TYPE=Collision, WEATHER=Heavy Rain, ROAD=Highway, TIME=Evening,
     PEAK=Yes, CONGESTION=Severe, LANES_BLOCKED=2+, SEVERITY=Serious}

and rules such as ``{WEATHER=Heavy Rain, ROAD=Highway} => {SEVERITY=Serious}``
are ranked by support, confidence, lift, leverage and conviction.
"""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import pandas as pd

Itemset = frozenset


# ------------------------------------------------------------------ Apriori
def apriori(transactions: list[set[str]], min_support: float,
            max_len: int = 3) -> dict[Itemset, float]:
    """Return every itemset whose support >= ``min_support``.

    Downward closure: an itemset can only be frequent if *all* of its subsets
    are frequent, so level k+1 candidates are built solely from level-k
    frequent itemsets and pruned before any scan of the database.
    """
    n = len(transactions)
    min_count = min_support * n
    frequent: dict[Itemset, float] = {}

    # L1 — single items.
    counts: dict[Itemset, int] = {}
    for t in transactions:
        for item in t:
            key = Itemset([item])
            counts[key] = counts.get(key, 0) + 1
    current = {k: c for k, c in counts.items() if c >= min_count}
    frequent.update({k: c / n for k, c in current.items()})

    k = 2
    while current and k <= max_len:
        candidates = _generate_candidates(set(current), k)
        if not candidates:
            break
        counts = {c: 0 for c in candidates}
        for t in transactions:                      # one database pass per level
            for c in candidates:
                if c <= t:
                    counts[c] += 1
        current = {c: n_ for c, n_ in counts.items() if n_ >= min_count}
        frequent.update({c: n_ / n for c, n_ in current.items()})
        k += 1
    return frequent


def _generate_candidates(prev_frequent: set[Itemset], k: int) -> set[Itemset]:
    """F(k-1) x F(k-1) join followed by the Apriori prune step."""
    candidates: set[Itemset] = set()
    prev = list(prev_frequent)
    for i in range(len(prev)):
        for j in range(i + 1, len(prev)):
            union = prev[i] | prev[j]
            if len(union) != k:
                continue
            # Prune: every (k-1)-subset must itself be frequent.
            if all(Itemset(sub) in prev_frequent for sub in combinations(union, k - 1)):
                candidates.add(union)
    return candidates


def generate_rules(frequent: dict[Itemset, float], min_confidence: float,
                   min_lift: float) -> pd.DataFrame:
    """Split every frequent itemset into antecedent => consequent pairs."""
    rows = []
    for itemset, support in frequent.items():
        if len(itemset) < 2:
            continue
        for r in range(1, len(itemset)):
            for antecedent in combinations(sorted(itemset), r):
                ante = Itemset(antecedent)
                cons = itemset - ante
                sup_a, sup_c = frequent.get(ante), frequent.get(cons)
                if not sup_a or not sup_c:
                    continue
                confidence = support / sup_a
                lift = confidence / sup_c
                if confidence < min_confidence or lift < min_lift:
                    continue
                rows.append({
                    "antecedent": ", ".join(sorted(ante)),
                    "consequent": ", ".join(sorted(cons)),
                    "support": round(support, 4),
                    "confidence": round(confidence, 4),
                    "lift": round(lift, 4),
                    "leverage": round(support - sup_a * sup_c, 4),
                    "conviction": round((1 - sup_c) / (1 - confidence), 4)
                    if confidence < 1 else float("inf"),
                    "antecedent_size": len(ante),
                })
    if not rows:
        return pd.DataFrame(columns=["antecedent", "consequent", "support", "confidence", "lift"])
    # Antecedent/consequent are included in the sort key so that ties — which are
    # common on categorical data — break deterministically across runs.
    return (pd.DataFrame(rows)
            .sort_values(["lift", "confidence", "support", "antecedent", "consequent"],
                         ascending=[False, False, False, True, True])
            .reset_index(drop=True))


# ------------------------------------------------------- transaction building
def build_transactions(incidents: pd.DataFrame) -> list[set[str]]:
    """Turn each incident row into a market-basket style transaction."""
    df = incidents
    lanes = pd.cut(df["lanes_blocked"], [-1, 0, 1, 99], labels=["0", "1", "2+"]).astype(str)
    veh = pd.cut(df["vehicles_involved"], [0, 1, 3, 99], labels=["1", "2-3", "4+"]).astype(str)

    items = pd.DataFrame({
        "type": "TYPE=" + df["incident_type"],
        "weather": "WEATHER=" + df["weather"],
        "road": "ROAD=" + df["road_type"],
        "zone": "ZONE=" + df["zone"],
        "tod": "TIME=" + df["time_of_day"],
        "peak": "PEAK=" + df["is_peak_hour"].map({1: "Yes", 0: "No"}),
        "weekend": "WEEKEND=" + df["is_weekend"].map({1: "Yes", 0: "No"}),
        "congestion": "CONGESTION=" + df["congestion_level"],
        "lanes": "LANES_BLOCKED=" + lanes,
        "vehicles": "VEHICLES=" + veh,
        "duration": "DURATION=" + df["duration_band"],
        "response": "RESPONSE=" + df["response_band"],
        "severity": "SEVERITY=" + df["severity_label"],
    })
    return [set(row) for row in items.itertuples(index=False, name=None)]


def run(incidents: pd.DataFrame, cfg: dict) -> dict:
    p = cfg["mining"]["apriori"]
    transactions = build_transactions(incidents)
    frequent = apriori(transactions, p["min_support"], p["max_itemset_size"])
    rules = generate_rules(frequent, p["min_confidence"], p["min_lift"])

    by_size: dict[str, int] = {}
    for its in frequent:
        by_size[f"L{len(its)}"] = by_size.get(f"L{len(its)}", 0) + 1

    severity_rules = rules[rules["consequent"].str.startswith("SEVERITY=")]

    print(f"[assoc] {len(transactions):,} transactions, "
          f"{len(frequent):,} frequent itemsets {by_size}")
    print(f"[assoc] {len(rules):,} rules pass support>={p['min_support']}, "
          f"conf>={p['min_confidence']}, lift>={p['min_lift']}")
    for _, r in rules.head(5).iterrows():
        print(f"[assoc]   {{{r['antecedent']}}} => {{{r['consequent']}}} "
              f"(sup={r['support']:.3f} conf={r['confidence']:.3f} lift={r['lift']:.2f})")

    top_frequent = sorted(frequent.items(), key=lambda kv: -kv[1])[:25]
    return {
        "n_transactions": len(transactions),
        "parameters": p,
        "frequent_itemsets_by_size": by_size,
        "n_rules": int(len(rules)),
        "top_rules": rules.head(25).to_dict(orient="records"),
        "severity_rules": severity_rules.head(15).to_dict(orient="records"),
        "top_frequent_itemsets": [
            {"itemset": ", ".join(sorted(k)), "support": round(v, 4)} for k, v in top_frequent
        ],
        "_rules": rules,
    }


def save(results: dict, out_dir: Path) -> Path:
    results["_rules"].to_csv(out_dir / "association_rules.csv", index=False)
    payload = {k: v for k, v in results.items() if not k.startswith("_")}
    path = out_dir / "association_results.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path

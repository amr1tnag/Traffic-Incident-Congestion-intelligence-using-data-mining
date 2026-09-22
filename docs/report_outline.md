# Capstone report outline

A chapter-by-chapter plan mapping the written report onto the code and the
generated artefacts. Numbers and tables are in `reports/findings.md`; figures
are in `reports/figures/`.

---

## 1. Introduction
* Motivation: urban congestion cost, incident response times, why a warehouse rather than ad-hoc scripts.
* Problem statement: turn raw detector and incident feeds into decisions — predict severity, locate hotspots, forecast congestion, surface unreported events.
* Scope and objectives (the five mining questions in the README table).
* Report organisation.

## 2. Literature / background
* KDD process (Fayyad et al.) and where each stage of this pipeline sits in it.
* Data warehousing: star vs. snowflake vs. fact constellation; why this project uses a constellation.
* OLAP operations and cuboid lattices.
* The mining families used here: decision trees & ensembles, k-Means/DBSCAN, Apriori, regression-based forecasting, isolation-based outlier detection.
* Related work on traffic-incident analytics; what this project adds.

## 3. System design
* Architecture diagram: sources → preprocessing → warehouse → OLAP → mining → reporting (`docs/architecture.md`).
* Warehouse schema: dimensions, fact tables, grain statement, hierarchies, measures (`src/warehouse_schema.sql`).
* Why the chosen grain (segment × hour) supports every downstream task.
* Materialised aggregates and their effect on query cost.
* Technology choices: Python, pandas, SQLite, scikit-learn — and why SQLite is sufficient at this scale.

## 4. Data
* The three source feeds, their columns and grain.
* Data generation model: demand profiles, weather penalties, incident capacity loss (`src/generate_data.py`). State plainly that data is simulated, and how the model is grounded.
* Injected data-quality defects and why they are there.
* Descriptive statistics — Figures 01–04.

## 5. Preprocessing
* Cleaning: duplicate removal, range validation, imputation strategy (segment × hour-of-week median, then column median), categorical unification.
* Integration: joining readings, incidents and reference data into the analytical base table.
* Transformation: calendar and cyclical features, ratios, discretisation of the congestion index.
* Reduction: the incident-level mining table.
* Quote the counts from `reports/preprocessing_report.json` — Section 2 of `findings.md`.

## 6. OLAP analysis
* One worked example per operation: roll-up, drill-down, slice, dice, pivot, cube (`src/olap.py`, `reports/olap/*.csv`).
* Findings: peak structure, road-class differences, adverse-weather effect, incident-hour versus normal-hour contrast.

## 7. Mining task 1 — Incident severity classification
* Feature set and target definition; the class imbalance and how it is handled.
* Four models, 5-fold CV on the training half, held-out test evaluation.
* Results table, confusion matrix, feature importances, extracted decision rules — Figure 05.
* Interpretation: which factors drive severity, and what dispatch policy that implies.

## 8. Mining task 2 — Hotspot clustering
* Segment profiling: the eight behavioural features and why each was chosen.
* Choosing k: elbow, silhouette, Davies–Bouldin, Calinski–Harabasz.
* Cluster archetypes and their operational meaning; DBSCAN's spatial view and its noise points — Figures 06–07.

## 9. Mining task 3 — Association rules
* Transaction encoding of an incident.
* Apriori walkthrough: candidate generation, the prune step, level-wise counting.
* Rule metrics and why lift/leverage matter beyond confidence.
* Top rules and what they tell the traffic authority — Figure 08.

## 10. Mining task 4 — Congestion forecasting
* Supervised framing of the time series; lag and rolling features.
* Chronological hold-out and why a random split would be invalid.
* Model comparison against persistence and seasonal-naive baselines — Figure 09.

## 11. Mining task 5 — Anomaly detection
* Isolation Forest and the seasonal z-score; what each is good at.
* Agreement analysis and anomalies without a logged incident — Figure 10.

## 12. Results and discussion
* Consolidated results table across all five tasks.
* What the analysis would change operationally: signal retiming, patrol placement, incident-response staging.
* Threats to validity: simulated data, single city, one year, no real-time latency model.

## 13. Conclusion and future work
* Recap against the objectives.
* Future work: real open-data ingestion, streaming ETL, spatio-temporal graph models, route-level impact simulation, a live dashboard.

## Appendices
* A — Warehouse DDL.
* B — Full OLAP query set.
* C — Complete results JSON.
* D — Test suite and how to reproduce every number.

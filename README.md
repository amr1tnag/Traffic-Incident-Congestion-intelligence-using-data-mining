# Traffic Incident & Congestion Intelligence

A complete **Data Warehousing and Data Mining** capstone: raw traffic feeds are
cleaned, loaded into a star-schema warehouse, sliced with OLAP operations, and
mined with five different techniques to answer operational questions a city
traffic authority actually asks.

```
 raw feeds ──▶ preprocessing ──▶ star-schema DW ──▶ OLAP cube ──▶ mining ──▶ report
 (3 sources)   (KDD cleaning)     (SQLite)          (5 ops)       (5 tasks)   (md + figures)
                                        │
                                        └──▶ CSV extract ──▶ Power BI (PBIP)
                                             (15 tables)      (6 pages)
```

## Questions the project answers

| # | Question | Technique |
|---|----------|-----------|
| 1 | How severe will a just-reported incident turn out to be? | **Classification** — decision tree, random forest, naive Bayes, logistic regression |
| 2 | Which road segments form congestion hotspots, and of what kind? | **Clustering** — k-Means (silhouette-selected k) + DBSCAN |
| 3 | Which conditions co-occur with serious incidents? | **Association rules** — Apriori implemented from scratch |
| 4 | What will congestion be over the next week? | **Regression / forecasting** — lag features + gradient boosting, vs. naive baselines |
| 5 | Which hours look abnormal and may be unreported events? | **Anomaly detection** — Isolation Forest + seasonal z-score |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python run_pipeline.py            # full run: generate -> ... -> report  (~2 minutes)
pytest -q                         # 11 unit tests
```

Outputs land in `reports/`:

| Path | Contents |
|------|----------|
| `reports/findings.md` | the full written findings, tables included |
| `reports/figures/*.png` | 10 figures for the report/viva slides |
| `reports/olap/*.csv` | every OLAP query result |
| `reports/*_results.json` | machine-readable metrics for each mining task |
| `data/warehouse/traffic_dw.db` | the loaded star-schema warehouse (queryable with any SQLite client) |

### Power BI

```bash
python run_pipeline.py --stages export   # refresh powerbi/data/ from the warehouse
python powerbi/build_pbip.py             # (re)generate the .pbip project
```

Open `powerbi/TrafficIntelligence.pbip` in Power BI Desktop and Refresh. The
model reuses the warehouse star schema unchanged — 16 tables, 35 DAX measures,
six report pages covering the overview, an interactive OLAP explorer, incidents,
hotspots, forecasts and a mining scorecard. Full notes in
[`docs/powerbi.md`](docs/powerbi.md).

Useful variations:

```bash
python run_pipeline.py --skip-generate             # reuse existing data/raw
python run_pipeline.py --stages preprocess etl olap
python -m src.olap                                 # just the cube operations
```

## Repository layout

```
config.yaml                 all tunable parameters in one place
run_pipeline.py             end-to-end driver (8 stages)
src/
  generate_data.py          stage 1 — raw source feeds (+ injected data-quality defects)
  preprocess.py             stage 2 — cleaning, integration, transformation, discretisation
  warehouse_schema.sql      star-schema DDL (4 dimensions, 2 fact tables)
  etl.py                    stage 3 — surrogate keys, fact loading, materialised aggregates
  olap.py                   stage 4 — roll-up, drill-down, slice, dice, pivot, cube
  mining/
    classification.py       severity prediction
    clustering.py           hotspot segmentation
    association.py          Apriori + rule generation (no library)
    forecasting.py          hourly congestion forecasting
    anomaly.py              outlier detection
  visualize.py              stage 6 — figures
  report.py                 stage 7 — findings.md
  export_bi.py              stage 8 — CSV extract for Power BI
powerbi/
  build_pbip.py             generates the Power BI project from the extract
  TrafficIntelligence.pbip  open this in Power BI Desktop
  *.SemanticModel/          TMDL model: tables, relationships, 35 DAX measures
  *.Report/                 PBIR report: 6 pages
tests/test_pipeline.py      unit tests for cleaning, Apriori, feature building
docs/                       architecture notes, report outline, Power BI guide
```

## The data

Three source feeds live in `data/raw/`:

| File | Grain | Key columns |
|------|-------|-------------|
| `road_segments.csv` | one row per segment | `segment_id, zone, road_type, lanes, length_km, speed_limit_kmph, lat, lon` |
| `traffic_readings.csv` | segment x hour | `vehicle_count, avg_speed_kmph, occupancy_pct, congestion_index, weather, ...` |
| `incidents.csv` | one row per incident | `incident_type, severity, vehicles_involved, lanes_blocked, duration_min, response_time_min` |

`generate_data.py` simulates one year (60 segments x 8 760 hours ≈ **525 600
segment-hours** and ~12 000 incidents) from documented traffic behaviour: twin
commuter peaks, a weekday/weekend split, weather penalties, and incidents that
reduce capacity and propagate congestion for as long as they last. Realistic
defects — duplicate transmissions, sensor dropouts, impossible values, two
vendors spelling weather differently — are injected on purpose so the
preprocessing stage is genuine work rather than a formality.

**Using real data instead:** drop CSVs with the same column names into
`data/raw/` and run `python run_pipeline.py --skip-generate`. Everything
downstream is source-agnostic. Public feeds that fit with light column
renaming include the UK STATS19 road-safety data, the US LSTW/countrywide
traffic-accident datasets, and most city open-data portals' loop-detector
exports.

## Warehouse design

Fact constellation (two fact tables over conformed dimensions):

```
        dim_time ─────┬──── fact_traffic ────┬───── dim_segment
                      │   (segment x hour)   │
     dim_weather ─────┤                      │
                      ├──── fact_incident ───┤
 dim_incident_type ───┘   (one per incident) └───── dim_segment
```

* **Grain** — `fact_traffic`: one row per segment per hour. `fact_incident`: one row per incident.
* **Hierarchies** — location `segment → zone → city`; time `hour → day → week → month → quarter → year`.
* **Measures** — congestion index, average speed, occupancy, travel time, delay index, vehicle-km, incident count, severity, duration, response time.
* **Aggregates** — `agg_zone_month`, `agg_roadtype_hour`, `agg_segment_daily` are materialised at load time so roll-up queries never rescan the base fact table.

## Method notes worth defending in the viva

* **No leakage in forecasting.** The hold-out is the chronologically final week, not a random split, and every feature is built from lags of the past. Persistence and seasonal-naive baselines are reported alongside the models so the improvement is honest.
* **Apriori is implemented, not imported.** Candidate generation is an F(k-1) × F(k-1) join followed by the downward-closure prune; one database pass per level. Rules are scored on support, confidence, lift, leverage and conviction — lift and leverage are what stop high-confidence-but-useless rules being reported.
* **Class imbalance.** Severity 1–4 is skewed, so classifiers use balanced class weights and are compared on **macro-F1**, not accuracy.
* **k is chosen, not assumed.** The elbow curve, silhouette, Davies–Bouldin and Calinski–Harabasz scores for every k in the range are saved in `reports/clustering_results.json`.
* **Two anomaly detectors, and their agreement.** A multivariate Isolation Forest and a seasonal z-score disagree in interesting ways; hours flagged by both *and* carrying no logged incident are the operationally interesting ones.

## Configuration

Everything tunable sits in `config.yaml` — simulation size, congestion bin
edges, outlier threshold, train/test split, the k range, Apriori thresholds,
and the forecast horizon. No parameter is hard-coded in the pipeline modules.

# Architecture

## Pipeline stages

| Stage | Module | Input | Output |
|-------|--------|-------|--------|
| 1. Generate | `src/generate_data.py` | `config.yaml` | `data/raw/*.csv` |
| 2. Preprocess | `src/preprocess.py` | raw CSVs | `data/processed/*.csv`, `reports/preprocessing_report.json` |
| 3. ETL | `src/etl.py` | processed CSVs | `data/warehouse/traffic_dw.db` |
| 4. OLAP | `src/olap.py` | warehouse | `reports/olap/*.csv` |
| 5. Mine | `src/mining/*` | analytical base + incident features | `reports/*_results.json`, supporting CSVs |
| 6. Visualise | `src/visualize.py` | mining results | `reports/figures/*.png` |
| 7. Report | `src/report.py` | everything above | `reports/findings.md` |

Each stage is independently runnable (`python -m src.<module>`), which is what
makes the pipeline debuggable — and makes it easy to demonstrate one stage at a
time in a viva.

## Data flow

```
road_segments.csv ─┐
traffic_readings.csv├─▶ clean ─▶ integrate ─▶ transform ─▶ analytical_base.csv ─┐
incidents.csv ─────┘                                  └─▶ incident_features.csv │
                                                                                │
                            ┌───────────────────────────────────────────────────┘
                            ▼
                     surrogate keys ─▶ dim_* + fact_* ─▶ agg_* (materialised)
                            │                                  │
                            ▼                                  ▼
                       mining tasks                       OLAP queries
                            └──────────▶ figures + findings.md ◀─────┘
```

## Star schema

```
                          ┌──────────────┐
                          │   dim_time   │  time_key (yyyymmddhh)
                          │  hour→day→   │  hour, day_name, week, month,
                          │  month→year  │  quarter, year, is_peak_hour
                          └──────┬───────┘
                                 │
┌───────────────┐         ┌──────▼────────┐        ┌────────────────┐
│  dim_weather  │◀────────│ fact_traffic  │───────▶│  dim_segment   │
│ weather,      │         │ (segment×hour)│        │ segment→zone→  │
│ is_adverse    │         │ congestion,   │        │ city; road_type│
└───────┬───────┘         │ speed, delay, │        │ lanes, geo     │
        │                 │ vehicle_km    │        └────────▲───────┘
        │                 └───────────────┘                 │
        │                 ┌───────────────┐                 │
        └────────────────▶│ fact_incident │─────────────────┘
                          │ severity,     │
                          │ duration,     │        ┌────────────────────┐
                          │ response_time │───────▶│ dim_incident_type  │
                          └───────────────┘        │ type → category    │
                                                   └────────────────────┘
```

Two fact tables sharing conformed dimensions makes this a **fact constellation
(galaxy) schema**. The shared `dim_time`, `dim_segment` and `dim_weather` are
what allow a single query to compare incident behaviour against the traffic
state on the same segment and hour.

## Design decisions

**Grain = segment × hour.** Fine enough for peak-structure analysis and
one-hour-ahead forecasting, coarse enough that a year of a 60-segment network
is half a million rows — comfortably handled by SQLite and by a laptop.

**Smart time key (`yyyymmddhh`).** Readable during debugging, sortable, and it
makes the time dimension joinable without a lookup while building facts.

**Surrogate keys on every other dimension.** Insulates the facts from changes
in natural keys upstream, as standard dimensional modelling requires.

**Aggregates materialised at load, not queried on demand.** `agg_zone_month`,
`agg_roadtype_hour` and `agg_segment_daily` are the pre-computed cuboids that
make roll-up interactive; they are rebuilt on every load, so they cannot drift.

**Mining reads the processed CSVs, not the warehouse.** The warehouse serves
OLAP; the flat analytical base serves mining. Keeping them separate avoids
tangling SQL aggregation with feature engineering, and either can be swapped
without touching the other.

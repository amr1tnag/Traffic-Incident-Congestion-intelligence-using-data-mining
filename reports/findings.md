# Traffic Incident & Congestion Intelligence — Findings

_Generated 2026-09-22 12:15 from the automated pipeline._

## 1. Dataset and warehouse

- Observation window: **2024-01-01 to 2024-12-30** (365 days, hourly)
- Road network: **60 segments** across 5 zones and 4 road classes
- Fact rows: **525,600** segment-hours and **12,218** incidents
- Warehouse: star schema at `traffic_dw.db` (4 dimensions, 2 fact tables, 3 materialised aggregates)

## 2. Data quality handled in preprocessing

| Issue | Records |
|---|---|
| Duplicate transmissions removed | 5,256 |
| Impossible sensor values voided | 4,729 |
| Missing values imputed | 25,245 |
| Extreme values flagged as outliers | 957 |
| Weather labels unified | 10,512 missing + case variants |

## 3. OLAP analysis

**Incident hours versus normal hours**

| state         |   observations |   avg_congestion |   avg_speed |   avg_travel_time |
|:--------------|---------------:|-----------------:|------------:|------------------:|
| Incident hour |          12218 |            0.729 |       24.47 |            13.797 |
| Normal hour   |         513382 |            0.349 |       38.95 |             6.641 |

**Ten most congested segments (roll-up over the year)**

| segment_id   | zone    | road_type   |   avg_congestion |   pct_congested_hours |   incidents |
|:-------------|:--------|:------------|-----------------:|----------------------:|------------:|
| SEG027       | North   | Arterial    |           0.4927 |                 38.11 |         299 |
| SEG022       | Central | Arterial    |           0.4914 |                 38.55 |         297 |
| SEG044       | Central | Arterial    |           0.4893 |                 37.11 |         328 |
| SEG029       | Central | Arterial    |           0.489  |                 38.08 |         265 |
| SEG030       | North   | Collector   |           0.4817 |                 36.46 |         213 |
| SEG046       | East    | Residential |           0.4795 |                 36.06 |         113 |
| SEG002       | Central | Arterial    |           0.4716 |                 33.95 |         278 |
| SEG034       | East    | Arterial    |           0.4599 |                 31.74 |         296 |
| SEG040       | South   | Collector   |           0.4561 |                 30.95 |         181 |
| SEG014       | South   | Residential |           0.4514 |                 30.21 |         119 |

**Congestion by road class and hour (pivot, selected hours)**

| road_type   |     3 |     8 |    12 |    18 |    22 |
|:------------|------:|------:|------:|------:|------:|
| Arterial    | 0.077 | 0.581 | 0.428 | 0.571 | 0.225 |
| Collector   | 0.077 | 0.589 | 0.432 | 0.576 | 0.226 |
| Highway     | 0.07  | 0.505 | 0.371 | 0.496 | 0.197 |
| Residential | 0.073 | 0.541 | 0.394 | 0.527 | 0.209 |

**Adverse-weather slice (Heavy Rain)**

| weather    | road_type   |   observations |   avg_congestion |   avg_delay_index |   pct_congested_hours |
|:-----------|:------------|---------------:|-----------------:|------------------:|----------------------:|
| Heavy Rain | Collector   |           8037 |           0.5008 |            0.4385 |                 32.56 |
| Heavy Rain | Arterial    |           8883 |           0.4979 |            0.4346 |                 30.5  |
| Heavy Rain | Residential |           4653 |           0.4706 |            0.4145 |                 25.08 |
| Heavy Rain | Highway     |           3807 |           0.4491 |            0.3978 |                 20.99 |

## 4. Incident severity classification

| model               |   cv_accuracy_mean |   test_accuracy |   macro_precision |   macro_recall |   macro_f1 |
|:--------------------|-------------------:|----------------:|------------------:|---------------:|-----------:|
| decision_tree       |             0.7052 |          0.7133 |            0.7202 |         0.744  |     0.7295 |
| random_forest       |             0.7428 |          0.7489 |            0.7684 |         0.7605 |     0.7642 |
| naive_bayes         |             0.5395 |          0.5339 |            0.53   |         0.5851 |     0.5388 |
| logistic_regression |             0.666  |          0.6691 |            0.66   |         0.6926 |     0.6725 |

Best model: **random_forest** (macro-F1 0.764, accuracy 0.749).

**Strongest predictors**

| feature                 |   importance |
|:------------------------|-------------:|
| flow_per_lane           |       0.1444 |
| congestion_index        |       0.109  |
| blocked_lane_ratio      |       0.0985 |
| incident_type_Collision |       0.095  |
| occupancy_pct           |       0.0765 |
| lanes_blocked           |       0.0746 |
| speed_ratio             |       0.0661 |
| avg_speed_kmph          |       0.0553 |
| vehicles_involved       |       0.0259 |
| hour                    |       0.024  |

**Decision tree (top levels)**

```
|--- congestion_index <= 0.90
|   |--- lanes_blocked <= -0.63
|   |   |--- incident_type_Collision <= 0.50
|   |   |   |--- congestion_index <= -0.18
|   |   |   |   |--- speed_limit_kmph <= 0.82
|   |   |   |   |   |--- truncated branch of depth 4
|   |   |   |   |--- speed_limit_kmph >  0.82
|   |   |   |   |   |--- truncated branch of depth 4
|   |   |   |--- congestion_index >  -0.18
|   |   |   |   |--- flow_per_lane <= 2.33
|   |   |   |   |   |--- truncated branch of depth 4
|   |   |   |   |--- flow_per_lane >  2.33
|   |   |   |   |   |--- truncated branch of depth 2
|   |   |--- incident_type_Collision >  0.50
|   |   |   |--- speed_ratio <= 0.92
|   |   |   |   |--- weather_Heavy Rain <= 0.50
|   |   |   |   |   |--- truncated branch of depth 4
|   |   |   |   |--- weather_Heavy Rain >  0.50
|   |   |   |   |   |--- class: 3
|   |   |   |--- speed_ratio >  0.92
|   |   |   |   |--- weather_Clear <= 0.50
|   |   |   |   |   |--- truncated branch of depth 4
|   |   |   |   |--- weather_Clear >  0.50
|   |   |   |   |   |--- truncated branch of depth 4
|   |--- lanes_blocked >  -0.63
|   |   |--- incident_type_Collision <= 0.50
|   |   |   |--- blocked_lane_ratio <= 0.11
|   |   |   |   |--- occupancy_pct <= -1.01
|   |   |   |   |   |--- truncated branch of depth 4
|   |   |   |   |--- occupancy_pct >  -1.01
```

## 5. Congestion hotspot clustering

k-Means selected **k = 2** by silhouette (0.371); DBSCAN found 7 dense spatial clusters and 24 noise segments.

|   cluster | label                        |   segments |   avg_congestion |   peak_congestion |   peak_offpeak_ratio |   incident_rate_per_1k_h |
|----------:|:-----------------------------|-----------:|-----------------:|------------------:|---------------------:|-------------------------:|
|         0 | Chronic peak-hour bottleneck |         26 |           0.4329 |            0.8395 |               1.9133 |                  25.72   |
|         1 | Stable low-stress segment    |         34 |           0.3003 |            0.5744 |               1.8997 |                  21.3538 |

**Top hotspot segments**

| segment_id   | zone    | road_type   |   avg_congestion |   peak_congestion |   incident_rate_per_1k_h | cluster_label                |
|:-------------|:--------|:------------|-----------------:|------------------:|-------------------------:|:-----------------------------|
| SEG027       | North   | Arterial    |           0.4927 |            0.9854 |                   34.132 | Chronic peak-hour bottleneck |
| SEG022       | Central | Arterial    |           0.4914 |            0.969  |                   33.904 | Chronic peak-hour bottleneck |
| SEG044       | Central | Arterial    |           0.4893 |            0.981  |                   37.443 | Chronic peak-hour bottleneck |
| SEG029       | Central | Arterial    |           0.489  |            0.9551 |                   30.251 | Chronic peak-hour bottleneck |
| SEG030       | North   | Collector   |           0.4817 |            0.936  |                   24.315 | Chronic peak-hour bottleneck |
| SEG046       | East    | Residential |           0.4795 |            0.9197 |                   12.9   | Chronic peak-hour bottleneck |
| SEG002       | Central | Arterial    |           0.4716 |            0.9134 |                   31.735 | Chronic peak-hour bottleneck |
| SEG034       | East    | Arterial    |           0.4599 |            0.8994 |                   33.79  | Chronic peak-hour bottleneck |
| SEG040       | South   | Collector   |           0.4561 |            0.8785 |                   20.662 | Chronic peak-hour bottleneck |
| SEG014       | South   | Residential |           0.4514 |            0.8587 |                   13.584 | Chronic peak-hour bottleneck |

## 6. Association rules (Apriori)

12,218 incident transactions; frequent itemsets {'L1': 51, 'L2': 568, 'L3': 1102}; **406 rules** at support ≥ 0.05, confidence ≥ 0.55, lift ≥ 1.1.

**Top rules by lift**

| antecedent                          | consequent                    |   support |   confidence |   lift |   leverage |
|:------------------------------------|:------------------------------|----------:|-------------:|-------:|-----------:|
| SEVERITY=Minor, TIME=Night          | CONGESTION=Free flow          |    0.0501 |       0.9415 | 6.0072 |     0.0418 |
| CONGESTION=Free flow, WEEKEND=No    | TIME=Night                    |    0.076  |       0.7638 | 4.9744 |     0.0607 |
| TIME=Night, WEATHER=Clear           | CONGESTION=Free flow          |    0.0584 |       0.7368 | 4.7012 |     0.046  |
| TIME=Night, VEHICLES=2-3            | CONGESTION=Free flow          |    0.064  |       0.7254 | 4.6283 |     0.0502 |
| TIME=Night                          | CONGESTION=Free flow, PEAK=No |    0.1102 |       0.718  | 4.6221 |     0.0864 |
| CONGESTION=Free flow, PEAK=No       | TIME=Night                    |    0.1102 |       0.7097 | 4.6221 |     0.0864 |
| PEAK=No, TIME=Night                 | CONGESTION=Free flow          |    0.1102 |       0.718  | 4.5811 |     0.0862 |
| TIME=Night                          | CONGESTION=Free flow          |    0.1102 |       0.718  | 4.5811 |     0.0862 |
| CONGESTION=Free flow                | PEAK=No, TIME=Night           |    0.1102 |       0.7034 | 4.5811 |     0.0862 |
| CONGESTION=Free flow                | TIME=Night                    |    0.1102 |       0.7034 | 4.5811 |     0.0862 |
| CONGESTION=Free flow, VEHICLES=2-3  | TIME=Night                    |    0.064  |       0.6988 | 4.5514 |     0.0499 |
| CONGESTION=Free flow, WEATHER=Clear | TIME=Night                    |    0.0584 |       0.6885 | 4.4842 |     0.0454 |

**Rules that conclude a severity level**

| antecedent                            | consequent     |   support |   confidence |   lift |
|:--------------------------------------|:---------------|----------:|-------------:|-------:|
| CONGESTION=Free flow, LANES_BLOCKED=0 | SEVERITY=Minor |    0.0622 |       0.7615 | 3.2854 |
| LANES_BLOCKED=0, TYPE=Breakdown       | SEVERITY=Minor |    0.0529 |       0.685  | 2.9555 |
| LANES_BLOCKED=0, WEATHER=Clear        | SEVERITY=Minor |    0.1032 |       0.6557 | 2.8291 |
| DURATION=20-45m, LANES_BLOCKED=0      | SEVERITY=Minor |    0.0697 |       0.6506 | 2.8069 |
| LANES_BLOCKED=0, PEAK=No              | SEVERITY=Minor |    0.1404 |       0.6451 | 2.7832 |
| CONGESTION=Light, LANES_BLOCKED=0     | SEVERITY=Minor |    0.067  |       0.6416 | 2.7679 |
| LANES_BLOCKED=0, VEHICLES=2-3         | SEVERITY=Minor |    0.1058 |       0.5832 | 2.5162 |
| LANES_BLOCKED=0, ROAD=Collector       | SEVERITY=Minor |    0.0619 |       0.5758 | 2.4841 |
| LANES_BLOCKED=0, ROAD=Arterial        | SEVERITY=Minor |    0.0729 |       0.5756 | 2.4832 |
| LANES_BLOCKED=0, RESPONSE=Slow        | SEVERITY=Minor |    0.05   |       0.5732 | 2.4728 |

## 7. Congestion forecasting

Chronological hold-out of the final 168 hours (8,424 training hours).

| model                   |     mae |    rmse |   mape_pct |     r2 |
|:------------------------|--------:|--------:|-----------:|-------:|
| persistence_baseline    | 0.06881 | 0.08696 |     24.24  | 0.8023 |
| seasonal_naive_baseline | 0.04252 | 0.06162 |     20.538 | 0.9007 |
| ridge                   | 0.03071 | 0.04052 |     12.972 | 0.9571 |
| random_forest           | 0.02257 | 0.03176 |      8.776 | 0.9736 |
| gradient_boosting       | 0.02279 | 0.03179 |      8.822 | 0.9736 |

Best model **random_forest** cuts MAE by **67.2%** against the persistence baseline.

## 8. Anomaly detection

- Isolation Forest flagged **5,080** of 525,600 segment-hours
- Seasonal z-score flagged **8,174**
- Both agreed on **1,598**, of which **326** have no logged incident — candidates for unreported events or sensor faults

| segment_id   | timestamp           | zone    | road_type   |   congestion_index |   avg_speed_kmph |   seasonal_z |   iso_score |   incident_count |
|:-------------|:--------------------|:--------|:------------|-------------------:|-----------------:|-------------:|------------:|-----------------:|
| SEG042       | 2024-07-13 09:00:00 | West    | Residential |                  1 |             3.21 |        5.477 |     -0.0961 |                1 |
| SEG042       | 2024-07-18 09:00:00 | West    | Residential |                  1 |             3    |        4.95  |     -0.0961 |                1 |
| SEG042       | 2024-01-25 18:00:00 | West    | Residential |                  1 |             3    |        4.821 |     -0.0959 |                1 |
| SEG023       | 2024-03-28 05:00:00 | Central | Residential |                  1 |             3.3  |        4.723 |     -0.0956 |                1 |
| SEG042       | 2024-09-29 08:00:00 | West    | Residential |                  1 |             3    |        5.105 |     -0.0951 |                1 |
| SEG014       | 2024-05-19 20:00:00 | South   | Residential |                  1 |             3    |        5.824 |     -0.0948 |                1 |
| SEG014       | 2024-06-05 12:00:00 | South   | Residential |                  1 |             3    |        3.767 |     -0.0948 |                1 |
| SEG014       | 2024-07-26 15:00:00 | South   | Residential |                  1 |             3    |        4.007 |     -0.0948 |                1 |
| SEG023       | 2024-01-30 05:00:00 | Central | Residential |                  1 |             3    |        6.203 |     -0.0948 |                1 |
| SEG023       | 2024-05-13 19:00:00 | Central | Residential |                  1 |             3    |        3.816 |     -0.0948 |                1 |

## 9. Figures

- `figures/01_congestion_profiles.png`
- `figures/02_weekly_heatmap.png`
- `figures/03_incident_overview.png`
- `figures/04_weather_impact.png`
- `figures/05_classification.png`
- `figures/06_clustering.png`
- `figures/07_hotspot_map.png`
- `figures/08_association_rules.png`
- `figures/09_forecast.png`
- `figures/10_anomalies.png`

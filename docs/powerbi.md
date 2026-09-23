# Power BI build

The same warehouse that drives `reports/findings.md` also drives a Power BI
model. Nothing is re-modelled: the star schema, the OLAP roll-ups and the five
mining outputs are exported once and consumed as-is.

```
 SQLite warehouse ──▶ src/export_bi.py ──▶ powerbi/data/*.csv ──▶ PBIP project
   (star schema)       (CSV extract)        (16 tables)           (model + report)
```

## Why not connect Power BI straight to SQLite?

Power BI Desktop has no native SQLite connector. The alternatives are an ODBC
driver (a per-machine install, 32/64-bit mismatches, and it breaks the moment
the file moves) or standing up a real server. A CSV extract needs neither, is
refreshable from a button, and keeps the whole thing portable — which is what
matters for a project that has to open on a marker's laptop.

## Build it

```bash
python run_pipeline.py                 # full pipeline, now ending in the export
# or, if the warehouse is already built:
python run_pipeline.py --stages export

python powerbi/build_pbip.py           # (re)generate the PBIP project
```

Then open `powerbi/TrafficIntelligence.pbip` in **Power BI Desktop** (December
2023 or later, with *Preview features → Power BI Project (.pbip) save option*
enabled) and hit **Refresh**.

One setting to change after cloning: **Transform data → Manage parameters →
`DataFolder`** must hold the absolute path to your `powerbi/data/` folder,
with a trailing slash — e.g. `C:/repo/powerbi/data/`. Re-running
`build_pbip.py` fills it in with the correct local path automatically, so in
practice you only edit it by hand if you move the folder.

## What gets built

**Semantic model** — 16 tables, 35 measures, 10 relationships:

| Group | Tables |
|---|---|
| Dimensions | `dim_date`, `dim_time`, `dim_segment`, `dim_weather`, `dim_incident_type` |
| Facts | `fact_traffic` (525,600 rows), `fact_incident` (12,218 rows) |
| Pre-aggregated cuboids | `agg_zone_month`, `agg_roadtype_hour`, `agg_segment_daily` (hidden) |
| Mining outputs | `mining_segment_clusters`, `mining_forecast`, `mining_anomalies`, `mining_association_rules`, `mining_model_metrics` |
| Measures | `_Measures` (a one-row calculated table that holds every DAX measure) |

`dim_date` exists only for Power BI. The warehouse's `dim_time` is hour-grain,
so its date column repeats 24 times a day and Power BI refuses to mark it as a
date table — which would cost you every time-intelligence function. `dim_date`
is a contiguous day-grain table sitting above it (`dim_date → dim_time →
facts`), so `DATEADD` and `DATESINPERIOD` work while the hourly grain stays
available for the congestion-profile visuals.

Hierarchies ship with the model: Year → Quarter → Month → Day on `dim_date`,
Zone → Road type → Segment on `dim_segment`, Category → Type on
`dim_incident_type`. Those give you drill-down and roll-up as clicks — the OLAP
operations the project implements in SQL, now interactive.

**Report** — six pages:

1. **Executive overview** — KPI cards, hourly congestion profile, zone and
   weather breakdowns, zone × month matrix.
2. **OLAP explorer** — the `road_type × hour` pivot and the zone → segment
   drill-down as live matrices, with weather / peak-hour / day / road-type
   slicers. This is where Power BI earns its place over the static CSVs.
3. **Incident analysis** — severity × weather matrix, incidents by type and by
   hour, response time vs clearance time per segment.
4. **Hotspots & clusters** — map coloured by k-Means cluster label, the cluster
   assignment table, most-congested segments.
5. **Forecast & anomalies** — actual vs predicted vs naive baseline over the
   168-hour horizon, anomalies by zone, and the flagged-hours table.
6. **Mining scorecard** — every model and metric from the five techniques in
   one matrix, plus the association rules.

The models stay in Python. Power BI visualises the scored outputs; it does not
re-run k-Means or gradient boosting in DAX, which it would do badly.

## Refreshing after a pipeline change

```bash
python run_pipeline.py --stages export   # rewrite powerbi/data/
```

Then **Refresh** in Desktop. Only re-run `build_pbip.py` if the *schema*
changed (a new column, table or measure) — it rewrites the model, so any
manual edits you made in Desktop to the model or report are overwritten.
Adding measures or visuals is best done by editing `build_pbip.py` and
regenerating, which keeps the project reproducible from source.

## Notes and limits

- `powerbi/data/` is gitignored — it is ~50 MB of derived CSV. The PBIP project
  *is* tracked, so a clone needs `--stages export` before its first refresh.
- `fact_traffic` is half a million rows: trivial for Import mode. DirectQuery
  is neither needed nor possible over CSV.
- SQLite has no boolean type, so `is_weekend`, `is_peak_hour`, `is_congested`
  and `is_major` arrive as 0/1 integers. The measures account for this
  (`CALCULATE(..., fact_traffic[is_congested] = 1)`), and the columns work
  directly as slicers.
- `discourageImplicitMeasures` is on: drag a measure onto a visual, not a raw
  fact column. That is deliberate — implicit sums over `congestion_index`
  are meaningless.
- The report is generated PBIR JSON and has not been opened in Desktop from
  this repo's CI. If a single visual fails to render, delete that visual's
  folder under `TrafficIntelligence.Report/definition/pages/<page>/visuals/`
  and rebuild it by hand; the semantic model is unaffected. If the whole report
  refuses to open, delete the `.Report` folder and start a new report against
  `TrafficIntelligence.SemanticModel` — the model is the part that carries the
  work.

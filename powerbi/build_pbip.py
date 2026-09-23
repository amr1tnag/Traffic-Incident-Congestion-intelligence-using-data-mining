#!/usr/bin/env python3
"""Generate the Power BI project (PBIP) for the traffic warehouse.

PBIP is Power BI Desktop's plain-text project format: the semantic model is
TMDL, the report is PBIR JSON.  Generating it instead of committing a binary
.pbix means the model is diffable, reviewable and rebuildable, and it stays in
step with the warehouse schema — re-run this after any schema change.

    python -m src.export_bi        # 1. dump the warehouse to powerbi/data/
    python powerbi/build_pbip.py   # 2. (re)generate the project around it

Then open powerbi/TrafficIntelligence.pbip in Power BI Desktop and refresh.
"""

from __future__ import annotations

import argparse
import json
import shutil
import uuid
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = "TrafficIntelligence"
NAMESPACE = uuid.UUID("6f1d4a3e-0b77-4c2a-9d51-8a2c7e5b1f04")

MEASURE_TABLE = "_Measures"

# Stable ids: the same table always gets the same lineage tag, so regenerating
# the project produces a clean diff instead of churning every GUID.
def gid(*parts: str) -> str:
    return str(uuid.uuid5(NAMESPACE, "/".join(parts)))


# --------------------------------------------------------------------- typing
# pandas dtype -> (TMDL dataType, Power Query type, summarizeBy)
def column_spec(table: str, name: str, dtype) -> tuple[str, str, str]:
    lowered = name.lower()
    if lowered in {"date"}:
        return "dateTime", "type date", "none"
    if lowered in {"timestamp", "exported_at"}:
        return "dateTime", "type datetime", "none"
    if pd.api.types.is_integer_dtype(dtype):
        summarize = "none" if _is_key_like(lowered) else "sum"
        return "int64", "Int64.Type", summarize
    if pd.api.types.is_float_dtype(dtype):
        summarize = "none" if lowered in {"latitude", "longitude"} else "sum"
        return "double", "type number", summarize
    return "string", "type text", "none"


def _is_key_like(name: str) -> bool:
    return (name.endswith("_key") or name.endswith("_id") or name == "cluster"
            or name.startswith("is_") or name in {
                "year", "quarter", "month", "week_of_year", "day_of_month",
                "day_of_week", "hour", "lanes", "speed_limit_kmph",
                "signalised", "severity", "dbscan_cluster"})


def format_string(table: str, name: str, data_type: str) -> str | None:
    lowered = name.lower()
    if data_type == "dateTime":
        return "yyyy-mm-dd hh:nn" if lowered == "timestamp" else "yyyy-mm-dd"
    if data_type == "double":
        if lowered in {"latitude", "longitude"}:
            return "0.00000"
        return "#,0.000"
    if data_type == "int64" and not _is_key_like(lowered):
        return "#,0"
    return None


def data_category(table: str, name: str) -> str | None:
    lowered = name.lower()
    if lowered == "latitude":
        return "Latitude"
    if lowered == "longitude":
        return "Longitude"
    if table == "dim_segment" and lowered == "zone":
        return "Place"
    return None


HIDDEN_COLUMNS = {
    ("dim_time", "time_key"), ("dim_time", "date_key"),
    ("dim_date", "date_key"),
    ("dim_segment", "segment_key"), ("dim_weather", "weather_key"),
    ("dim_incident_type", "incident_type_key"),
    ("fact_traffic", "traffic_key"), ("fact_traffic", "time_key"),
    ("fact_traffic", "segment_key"), ("fact_traffic", "weather_key"),
    ("fact_incident", "incident_key"), ("fact_incident", "time_key"),
    ("fact_incident", "segment_key"), ("fact_incident", "weather_key"),
    ("fact_incident", "incident_type_key"),
}

# Fact tables are hidden from the field list: users pick measures, not raw
# columns.  This is what `discourageImplicitMeasures` is for.
HIDDEN_TABLES = {"agg_zone_month", "agg_roadtype_hour", "agg_segment_daily"}


# -------------------------------------------------------------- relationships
# (name, fromTable, fromColumn, toTable, toColumn)
RELATIONSHIPS = [
    ("dim_date_dim_time", "dim_time", "date_key", "dim_date", "date_key"),
    ("dim_time_fact_traffic", "fact_traffic", "time_key", "dim_time", "time_key"),
    ("dim_segment_fact_traffic", "fact_traffic", "segment_key", "dim_segment", "segment_key"),
    ("dim_weather_fact_traffic", "fact_traffic", "weather_key", "dim_weather", "weather_key"),
    ("dim_time_fact_incident", "fact_incident", "time_key", "dim_time", "time_key"),
    ("dim_segment_fact_incident", "fact_incident", "segment_key", "dim_segment", "segment_key"),
    ("dim_weather_fact_incident", "fact_incident", "weather_key", "dim_weather", "weather_key"),
    ("dim_type_fact_incident", "fact_incident", "incident_type_key",
     "dim_incident_type", "incident_type_key"),
    # Mining outputs join on the natural key, not the surrogate one.
    ("segment_clusters", "mining_segment_clusters", "segment_id", "dim_segment", "segment_id"),
    ("anomalies_segment", "mining_anomalies", "segment_id", "dim_segment", "segment_id"),
]

# ------------------------------------------------------------------ hierarchies
HIERARCHIES = {
    "dim_date": [("Calendar", [("Year", "year"), ("Quarter", "quarter_name"),
                               ("Month", "month_name"), ("Day", "day_of_month")])],
    "dim_time": [("Time of day", [("Time of day", "time_of_day"), ("Hour", "hour")])],
    "dim_segment": [("Geography", [("Zone", "zone"), ("Road type", "road_type"),
                                   ("Segment", "segment_id")])],
    "dim_incident_type": [("Incident", [("Category", "category"),
                                        ("Type", "incident_type")])],
}

# ---------------------------------------------------------------------- measures
# (name, DAX, formatString, description)
MEASURES = [
    # --- traffic volume and congestion
    ("Observations", "COUNTROWS(fact_traffic)", "#,0",
     "Segment-hours in the current filter context."),
    ("Avg Congestion", "AVERAGE(fact_traffic[congestion_index])", "0.000",
     "Mean congestion index (0 = free flow, 1 = gridlock)."),
    ("Peak Congestion", "MAX(fact_traffic[congestion_index])", "0.000",
     "Worst congestion index observed."),
    ("Avg Speed", "AVERAGE(fact_traffic[avg_speed_kmph])", "#,0.0",
     "Mean observed speed in km/h."),
    ("Avg Speed Ratio", "AVERAGE(fact_traffic[speed_ratio])", "0.000",
     "Observed speed as a fraction of the posted limit."),
    ("Avg Travel Time", "AVERAGE(fact_traffic[travel_time_min])", "#,0.0",
     "Mean segment traversal time in minutes."),
    ("Avg Delay Index", "AVERAGE(fact_traffic[delay_index])", "0.000",
     "Excess travel time relative to free-flow conditions."),
    ("Vehicle KM", "SUM(fact_traffic[vehicle_km])", "#,0",
     "Total vehicle-kilometres travelled."),
    ("Congested Hours",
     "CALCULATE(COUNTROWS(fact_traffic), fact_traffic[is_congested] = 1)", "#,0",
     "Segment-hours classified as congested."),
    ("Congested Hours %",
     "DIVIDE([Congested Hours], [Observations])", "0.0%",
     "Share of segment-hours that were congested."),
    # --- incidents
    ("Incidents", "COUNTROWS(fact_incident)", "#,0",
     "Reported incidents in the current filter context."),
    ("Major Incidents",
     "CALCULATE(COUNTROWS(fact_incident), fact_incident[is_major] = 1)", "#,0",
     "Incidents flagged as major (severity 3-4)."),
    ("Major Incident %", "DIVIDE([Major Incidents], [Incidents])", "0.0%",
     "Share of incidents that were major."),
    ("Avg Severity", "AVERAGE(fact_incident[severity])", "0.00",
     "Mean reported severity, 1 (minor) to 4 (critical)."),
    ("Avg Response Time", "AVERAGE(fact_incident[response_time_min])", "#,0.0",
     "Mean minutes from report to first response."),
    ("Avg Incident Duration", "AVERAGE(fact_incident[duration_min])", "#,0.0",
     "Mean incident clearance time in minutes."),
    ("Casualties", "SUM(fact_incident[casualties])", "#,0",
     "Total casualties recorded."),
    ("Incidents per 1k Segment-Hours",
     "DIVIDE([Incidents], [Observations]) * 1000", "#,0.00",
     "Incident rate normalised for network exposure."),
    # --- time intelligence (needs dim_date marked as the date table)
    ("Avg Congestion PM",
     "CALCULATE([Avg Congestion], DATEADD(dim_date[date], -1, MONTH))", "0.000",
     "Average congestion in the previous month."),
    ("Congestion MoM %",
     "VAR Prev = [Avg Congestion PM]\n"
     "RETURN DIVIDE([Avg Congestion] - Prev, Prev)", "+0.0%;-0.0%;0.0%",
     "Month-on-month change in average congestion."),
    ("Incidents PM",
     "CALCULATE([Incidents], DATEADD(dim_date[date], -1, MONTH))", "#,0",
     "Incidents in the previous month."),
    ("Incidents MoM %",
     "VAR Prev = [Incidents PM]\n"
     "RETURN DIVIDE([Incidents] - Prev, Prev)", "+0.0%;-0.0%;0.0%",
     "Month-on-month change in incident count."),
    ("Avg Congestion 7d",
     "AVERAGEX(\n"
     "    DATESINPERIOD(dim_date[date], MAX(dim_date[date]), -7, DAY),\n"
     "    [Avg Congestion]\n"
     ")", "0.000",
     "Seven-day rolling mean congestion."),
    # --- weather and peak contrasts, i.e. the OLAP slice/dice questions as DAX
    ("Avg Congestion Adverse Weather",
     "CALCULATE([Avg Congestion], dim_weather[is_adverse] = 1)", "0.000",
     "Average congestion while weather is adverse."),
    ("Wet Weather Penalty",
     "VAR Clear = CALCULATE([Avg Congestion], dim_weather[is_adverse] = 0)\n"
     "VAR Adverse = CALCULATE([Avg Congestion], dim_weather[is_adverse] = 1)\n"
     "RETURN Adverse - Clear", "+0.000;-0.000;0.000",
     "Extra congestion attributable to adverse weather."),
    ("Peak vs Offpeak Ratio",
     "VAR Peak = CALCULATE([Avg Congestion], dim_time[is_peak_hour] = 1)\n"
     "VAR Offpeak = CALCULATE([Avg Congestion], dim_time[is_peak_hour] = 0)\n"
     "RETURN DIVIDE(Peak, Offpeak)", "#,0.00",
     "How much worse the peak is than the off-peak."),
    # --- mining satellites
    ("Anomalies", "COUNTROWS(mining_anomalies)", "#,0",
     "Hours flagged by the anomaly ensemble."),
    ("Agreed Anomalies",
     "CALCULATE(COUNTROWS(mining_anomalies), mining_anomalies[anomaly_agreement] = 1)",
     "#,0", "Hours flagged by both Isolation Forest and the seasonal z-score."),
    ("Unexplained Anomalies",
     "CALCULATE(\n"
     "    COUNTROWS(mining_anomalies),\n"
     "    mining_anomalies[anomaly_agreement] = 1,\n"
     "    mining_anomalies[incident_count] = 0\n"
     ")", "#,0",
     "Agreed anomalies with no logged incident - candidate unreported events."),
    ("Forecast MAE",
     "AVERAGEX(mining_forecast, ABS(mining_forecast[actual] - mining_forecast[predicted]))",
     "0.0000", "Mean absolute error of the congestion forecast."),
    ("Persistence MAE",
     "AVERAGEX(mining_forecast, ABS(mining_forecast[actual] - mining_forecast[persistence]))",
     "0.0000", "Mean absolute error of the naive persistence baseline."),
    ("Forecast Lift vs Persistence %",
     "DIVIDE([Persistence MAE] - [Forecast MAE], [Persistence MAE])", "0.0%",
     "How much the model beats the naive baseline."),
    ("Rules", "COUNTROWS(mining_association_rules)", "#,0",
     "Association rules passing the support/confidence/lift thresholds."),
    ("Max Lift", "MAX(mining_association_rules[lift])", "#,0.00",
     "Strongest lift among the selected rules."),
    ("Segments Clustered", "COUNTROWS(mining_segment_clusters)", "#,0",
     "Road segments assigned to a congestion-profile cluster."),
]


# ============================================================ TMDL generation
def tmdl_expression(dax: str, indent: str) -> str:
    """Render a DAX/M expression, using TMDL's triple-backtick block if multiline."""
    if "\n" not in dax:
        return f" = {dax}"
    body = "\n".join(f"{indent}\t\t\t{line}" if line else ""
                     for line in dax.splitlines())
    return f" =\n{indent}\t\t\t```\n{body}\n{indent}\t\t\t```"


def power_query(table: str, columns: list[tuple[str, str]]) -> str:
    """M query reading one exported CSV and typing every column explicitly.

    Explicit typing matters: left to its own devices Power Query samples the
    first 200 rows, which turns sparse integer columns into text and silently
    breaks the measures that sum them.
    """
    types = ", ".join(f'{{"{name}", {pq_type}}}' for name, pq_type in columns)
    return (
        'let\n'
        f'    Source = Csv.Document(\n'
        f'        File.Contents(DataFolder & "{table}.csv"),\n'
        '        [Delimiter = ",", Encoding = 65001, QuoteStyle = QuoteStyle.Csv]\n'
        '    ),\n'
        '    Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars = true]),\n'
        f'    Typed = Table.TransformColumnTypes(Promoted, {{{types}}})\n'
        'in\n'
        '    Typed'
    )


def table_tmdl(table: str, frame: pd.DataFrame) -> str:
    lines = [f"table {table}", f"\tlineageTag: {gid('table', table)}", ""]
    if table in HIDDEN_TABLES:
        lines.insert(1, "\tisHidden")
    if table == "dim_date":
        # Marks this as *the* date table, enabling DATEADD / DATESINPERIOD.
        lines.insert(1, "\tdataCategory: Time")

    pq_columns: list[tuple[str, str]] = []
    for name in frame.columns:
        data_type, pq_type, summarize = column_spec(table, name, frame[name].dtype)
        pq_columns.append((name, pq_type))

        lines.append(f"\tcolumn {quote(name)}")
        lines.append(f"\t\tdataType: {data_type}")
        if (table, name) in HIDDEN_COLUMNS:
            lines.append("\t\tisHidden")
        if table == "dim_date" and name == "date":
            lines.append("\t\tisKey")
        fmt = format_string(table, name, data_type)
        if fmt:
            lines.append(f"\t\tformatString: {fmt}")
        lines.append(f"\t\tlineageTag: {gid('column', table, name)}")
        lines.append(f"\t\tsummarizeBy: {summarize}")
        lines.append(f"\t\tsourceColumn: {name}")
        category = data_category(table, name)
        if category:
            lines.append(f"\t\tdataCategory: {category}")
        if name.lower() in {"month_name", "day_name"}:
            # Sort January before April rather than alphabetically.
            sort_by = "month" if name.lower() == "month_name" else "day_of_week"
            if sort_by in frame.columns:
                lines.append(f"\t\tsortByColumn: {sort_by}")
        lines.append("")
        lines.append("\t\tannotation SummarizationSetBy = Automatic")
        lines.append("")

    for hierarchy, levels in HIERARCHIES.get(table, []):
        available = [(lvl, col) for lvl, col in levels if col in frame.columns]
        if not available:
            continue
        lines.append(f"\thierarchy {quote(hierarchy)}")
        lines.append(f"\t\tlineageTag: {gid('hierarchy', table, hierarchy)}")
        lines.append("")
        for ordinal, (level, column) in enumerate(available):
            lines.append(f"\t\tlevel {quote(level)}")
            lines.append(
                f"\t\t\tlineageTag: {gid('level', table, hierarchy, level)}")
            lines.append(f"\t\t\tcolumn: {quote(column)}")
            lines.append("")

    lines.append(f"\tpartition {quote(table)} = m")
    lines.append("\t\tmode: import")
    lines.append("\t\tsource =")
    for line in power_query(table, pq_columns).splitlines():
        lines.append(f"\t\t\t\t{line}" if line else "")
    lines.append("")
    lines.append("\tannotation PBI_ResultType = Table")
    lines.append("")
    return "\n".join(lines)


def measures_tmdl() -> str:
    lines = [f"table {quote(MEASURE_TABLE)}",
             f"\tlineageTag: {gid('table', MEASURE_TABLE)}", ""]
    for name, dax, fmt, description in MEASURES:
        lines.append(f"\t/// {description}")
        lines.append(f"\tmeasure {quote(name)}{tmdl_expression(dax, chr(9))}")
        lines.append(f"\t\tformatString: {fmt}")
        lines.append(f"\t\tlineageTag: {gid('measure', name)}")
        lines.append("")
    # A calculated one-row table gives the measures a home without importing
    # anything; the column itself is hidden so the folder shows measures only.
    lines.append("\tcolumn Value")
    lines.append("\t\tdataType: int64")
    lines.append("\t\tisHidden")
    lines.append(f"\t\tlineageTag: {gid('column', MEASURE_TABLE, 'Value')}")
    lines.append("\t\tsummarizeBy: none")
    lines.append("\t\tsourceColumn: [Value]")
    lines.append("")
    lines.append("\t\tannotation SummarizationSetBy = Automatic")
    lines.append("")
    lines.append(f"\tpartition {quote(MEASURE_TABLE)} = calculated")
    lines.append("\t\tmode: import")
    lines.append("\t\tsource = {1}")
    lines.append("")
    return "\n".join(lines)


def relationships_tmdl(tables: set[str]) -> str:
    blocks = []
    for name, from_table, from_col, to_table, to_col in RELATIONSHIPS:
        if from_table not in tables or to_table not in tables:
            continue
        blocks.append(
            f"relationship {name}\n"
            f"\tfromColumn: {from_table}.{quote(from_col)}\n"
            f"\ttoColumn: {to_table}.{quote(to_col)}\n"
        )
    return "\n".join(blocks)


def quote(name: str) -> str:
    """TMDL needs quoting for identifiers that are not bare words."""
    return f"'{name}'" if not name.replace("_", "").isalnum() else name


# ========================================================== PBIR report layout
PAGE_W, PAGE_H = 1280, 720

SCHEMA = "https://developer.microsoft.com/json-schemas/fabric/item/report/definition"


def col(table: str, name: str) -> dict:
    return {"Column": {"Expression": {"SourceRef": {"Entity": table}},
                       "Property": name}}


def msr(name: str) -> dict:
    return {"Measure": {"Expression": {"SourceRef": {"Entity": MEASURE_TABLE}},
                        "Property": name}}


def projection(field: dict) -> dict:
    if "Column" in field:
        entity = field["Column"]["Expression"]["SourceRef"]["Entity"]
        prop = field["Column"]["Property"]
    else:
        entity = field["Measure"]["Expression"]["SourceRef"]["Entity"]
        prop = field["Measure"]["Property"]
    return {"field": field, "queryRef": f"{entity}.{prop}", "nativeQueryRef": prop}


def visual(name: str, visual_type: str, x: int, y: int, w: int, h: int,
           roles: dict[str, list[dict]] | None = None,
           title: str | None = None, order: int = 0,
           objects: dict | None = None) -> dict:
    """One PBIR visual container.  `roles` maps a data role to its fields."""
    spec: dict = {
        "$schema": f"{SCHEMA}/visualContainer/1.0.0/schema.json",
        "name": gid("visual", name),
        "position": {"x": x, "y": y, "z": order, "width": w, "height": h,
                     "tabOrder": order},
        "visual": {"visualType": visual_type, "drillFilterOtherVisuals": True},
        "slug": name,
    }
    if roles:
        spec["visual"]["query"] = {
            "queryState": {
                role: {"projections": [projection(f) for f in fields]}
                for role, fields in roles.items()
            }
        }
    visual_objects = dict(objects or {})
    if title:
        visual_objects["title"] = [{
            "properties": {
                "text": {"expr": {"Literal": {"Value": f"'{title}'"}}},
                "show": {"expr": {"Literal": {"Value": "true"}}},
            }
        }]
    if visual_objects:
        spec["visual"]["objects"] = visual_objects
    return spec


def textbox(name: str, text: str, x: int, y: int, w: int, h: int,
            size: int = 20, order: int = 0) -> dict:
    """A heading.  Textboxes carry their content in objects, not a query."""
    return {
        "$schema": f"{SCHEMA}/visualContainer/1.0.0/schema.json",
        "name": gid("visual", name),
        "position": {"x": x, "y": y, "z": order, "width": w, "height": h,
                     "tabOrder": order},
        "visual": {
            "visualType": "textbox",
            "objects": {
                "general": [{
                    "properties": {
                        "paragraphs": [{
                            "textRuns": [{
                                "value": text,
                                "textStyle": {"fontSize": f"{size}pt",
                                              "fontWeight": "bold"},
                            }]
                        }]
                    }
                }]
            },
        },
        "slug": name,
    }


def card(name: str, measure: str, x: int, y: int, order: int,
         w: int = 200, h: int = 110) -> dict:
    return visual(name, "card", x, y, w, h,
                  roles={"Values": [msr(measure)]}, title=measure, order=order)


def build_pages() -> list[tuple[str, str, list[dict]]]:
    """(page id, display name, visuals) for every report page."""
    pages: list[tuple[str, str, list[dict]]] = []

    # ---------------------------------------------------- 1. executive overview
    v = [
        textbox("ov_title", "Traffic Incident & Congestion Intelligence", 20, 16, 760, 44),
        card("ov_c1", "Avg Congestion", 20, 72, 1),
        card("ov_c2", "Congested Hours %", 232, 72, 2),
        card("ov_c3", "Incidents", 444, 72, 3),
        card("ov_c4", "Avg Response Time", 656, 72, 4),
        card("ov_c5", "Avg Speed", 868, 72, 5),
        visual("ov_hour", "lineChart", 20, 198, 620, 250,
               roles={"Category": [col("dim_time", "hour")],
                      "Y": [msr("Avg Congestion"), msr("Avg Speed Ratio")]},
               title="Congestion profile by hour of day", order=6),
        visual("ov_zone", "clusteredColumnChart", 660, 198, 400, 250,
               roles={"Category": [col("dim_segment", "zone")],
                      "Y": [msr("Avg Congestion")]},
               title="Average congestion by zone", order=7),
        visual("ov_matrix", "pivotTable", 20, 462, 620, 238,
               roles={"Rows": [col("dim_segment", "zone")],
                      "Columns": [col("dim_date", "month_name")],
                      "Values": [msr("Avg Congestion")]},
               title="Zone x month roll-up", order=8),
        visual("ov_weather", "clusteredBarChart", 660, 462, 400, 238,
               roles={"Category": [col("dim_weather", "weather")],
                      "Y": [msr("Avg Congestion")]},
               title="Congestion by weather", order=9),
        visual("ov_slicer_zone", "slicer", 1076, 198, 184, 240,
               roles={"Values": [col("dim_segment", "zone")]},
               title="Zone", order=10),
        visual("ov_slicer_month", "slicer", 1076, 462, 184, 238,
               roles={"Values": [col("dim_date", "month_name")]},
               title="Month", order=11),
    ]
    pages.append(("executive-overview", "1. Executive overview", v))

    # -------------------------------------------------------- 2. OLAP explorer
    v = [
        textbox("ol_title", "OLAP explorer - slice, dice, roll up, drill down",
                20, 16, 760, 40, size=18),
        visual("ol_pivot", "pivotTable", 20, 68, 840, 330,
               roles={"Rows": [col("dim_segment", "road_type")],
                      "Columns": [col("dim_time", "hour")],
                      "Values": [msr("Avg Congestion")]},
               title="Road type x hour (the pivot cuboid, now interactive)", order=1),
        visual("ol_rollup", "pivotTable", 20, 410, 840, 290,
               roles={"Rows": [col("dim_segment", "zone"),
                               col("dim_segment", "segment_id")],
                      "Values": [msr("Avg Congestion"), msr("Avg Speed"),
                                 msr("Congested Hours %"), msr("Incidents")]},
               title="Zone -> segment drill-down", order=2),
        visual("ol_s_weather", "slicer", 880, 68, 180, 200,
               roles={"Values": [col("dim_weather", "weather")]},
               title="Weather", order=3),
        visual("ol_s_peak", "slicer", 1076, 68, 184, 200,
               roles={"Values": [col("dim_time", "is_peak_hour")]},
               title="Peak hour", order=4),
        visual("ol_s_day", "slicer", 880, 284, 180, 200,
               roles={"Values": [col("dim_date", "day_name")]},
               title="Day", order=5),
        visual("ol_s_road", "slicer", 1076, 284, 184, 200,
               roles={"Values": [col("dim_segment", "road_type")]},
               title="Road type", order=6),
        card("ol_c1", "Peak vs Offpeak Ratio", 880, 500, 7),
        card("ol_c2", "Wet Weather Penalty", 1092, 500, 8, w=168),
        card("ol_c3", "Vehicle KM", 880, 620, 9),
        card("ol_c4", "Observations", 1092, 620, 10, w=168),
    ]
    pages.append(("olap-explorer", "2. OLAP explorer", v))

    # ---------------------------------------------------- 3. incident analysis
    v = [
        textbox("in_title", "Incident analysis", 20, 16, 600, 40, size=18),
        card("in_c1", "Incidents", 20, 64, 1),
        card("in_c2", "Major Incident %", 232, 64, 2),
        card("in_c3", "Avg Severity", 444, 64, 3),
        card("in_c4", "Casualties", 656, 64, 4),
        card("in_c5", "Avg Incident Duration", 868, 64, 5),
        visual("in_sev_weather", "pivotTable", 20, 190, 500, 250,
               roles={"Rows": [col("dim_weather", "weather")],
                      "Columns": [col("fact_incident", "severity")],
                      "Values": [msr("Incidents")]},
               title="Severity x weather", order=6),
        visual("in_type", "clusteredBarChart", 540, 190, 480, 250,
               roles={"Category": [col("dim_incident_type", "incident_type")],
                      "Y": [msr("Incidents")]},
               title="Incidents by type", order=7),
        visual("in_scatter", "scatterChart", 20, 452, 500, 248,
               roles={"X": [msr("Avg Response Time")],
                      "Y": [msr("Avg Incident Duration")],
                      "Category": [col("dim_segment", "segment_id")]},
               title="Response time vs clearance time by segment", order=8),
        visual("in_hour", "columnChart", 540, 452, 480, 248,
               roles={"Category": [col("dim_time", "hour")],
                      "Y": [msr("Incidents")]},
               title="Incidents by hour", order=9),
        visual("in_s_cat", "slicer", 1040, 190, 220, 250,
               roles={"Values": [col("dim_incident_type", "category")]},
               title="Category", order=10),
        visual("in_s_zone", "slicer", 1040, 452, 220, 248,
               roles={"Values": [col("dim_segment", "zone")]},
               title="Zone", order=11),
    ]
    pages.append(("incident-analysis", "3. Incident analysis", v))

    # ------------------------------------------------- 4. hotspots & clusters
    v = [
        textbox("hs_title", "Congestion hotspots and segment clusters",
                20, 16, 700, 40, size=18),
        visual("hs_map", "map", 20, 64, 620, 380,
               roles={"Category": [col("dim_segment", "segment_id")],
                      "Series": [col("mining_segment_clusters", "cluster_label")],
                      "Size": [msr("Avg Congestion")]},
               title="Hotspot map (bubble size = average congestion)", order=1),
        visual("hs_table", "tableEx", 660, 64, 600, 380,
               roles={"Values": [col("mining_segment_clusters", "segment_id"),
                                 col("mining_segment_clusters", "cluster_label"),
                                 col("mining_segment_clusters", "avg_congestion"),
                                 col("mining_segment_clusters", "pct_congested_hours"),
                                 col("mining_segment_clusters", "peak_offpeak_ratio"),
                                 col("mining_segment_clusters", "incident_rate_per_1k_h")]},
               title="Cluster assignment per segment", order=2),
        visual("hs_top", "clusteredBarChart", 20, 456, 620, 244,
               roles={"Category": [col("dim_segment", "segment_id")],
                      "Y": [msr("Avg Congestion")]},
               title="Most congested segments", order=3),
        visual("hs_profile", "clusteredColumnChart", 660, 456, 400, 244,
               roles={"Category": [col("mining_segment_clusters", "cluster_label")],
                      "Y": [msr("Segments Clustered")]},
               title="Segments per cluster", order=4),
        visual("hs_s_cluster", "slicer", 1076, 456, 184, 244,
               roles={"Values": [col("mining_segment_clusters", "cluster_label")]},
               title="Cluster", order=5),
    ]
    pages.append(("hotspots-clusters", "4. Hotspots & clusters", v))

    # ------------------------------------------------- 5. forecast & anomalies
    v = [
        textbox("fa_title", "Forecast and anomaly detection", 20, 16, 700, 40, size=18),
        card("fa_c1", "Forecast MAE", 20, 64, 1),
        card("fa_c2", "Persistence MAE", 232, 64, 2),
        card("fa_c3", "Forecast Lift vs Persistence %", 444, 64, 3, w=236),
        card("fa_c4", "Agreed Anomalies", 692, 64, 4),
        card("fa_c5", "Unexplained Anomalies", 904, 64, 5, w=236),
        visual("fa_forecast", "lineChart", 20, 190, 1240, 260,
               roles={"Category": [col("mining_forecast", "timestamp")],
                      "Y": [col("mining_forecast", "actual"),
                            col("mining_forecast", "predicted"),
                            col("mining_forecast", "persistence")]},
               title="One-week ahead congestion: actual vs predicted vs naive baseline",
               order=6),
        visual("fa_anom_zone", "clusteredColumnChart", 20, 462, 400, 238,
               roles={"Category": [col("mining_anomalies", "zone")],
                      "Y": [msr("Agreed Anomalies")]},
               title="Agreed anomalies by zone", order=7),
        visual("fa_anom_table", "tableEx", 440, 462, 620, 238,
               roles={"Values": [col("mining_anomalies", "segment_id"),
                                 col("mining_anomalies", "timestamp"),
                                 col("mining_anomalies", "congestion_index"),
                                 col("mining_anomalies", "seasonal_z"),
                                 col("mining_anomalies", "incident_count")]},
               title="Flagged hours", order=8),
        visual("fa_s_zone", "slicer", 1076, 462, 184, 238,
               roles={"Values": [col("mining_anomalies", "zone")]},
               title="Zone", order=9),
    ]
    pages.append(("forecast-anomalies", "5. Forecast & anomalies", v))

    # --------------------------------------------------- 6. mining scorecard
    v = [
        textbox("ms_title", "Data mining scorecard", 20, 16, 700, 40, size=18),
        visual("ms_matrix", "pivotTable", 20, 64, 840, 400,
               roles={"Rows": [col("mining_model_metrics", "task"),
                               col("mining_model_metrics", "model")],
                      "Columns": [col("mining_model_metrics", "metric")],
                      "Values": [col("mining_model_metrics", "value")]},
               title="Every model, every metric", order=1),
        visual("ms_rules", "tableEx", 20, 476, 840, 224,
               roles={"Values": [col("mining_association_rules", "antecedent"),
                                 col("mining_association_rules", "consequent"),
                                 col("mining_association_rules", "support"),
                                 col("mining_association_rules", "confidence"),
                                 col("mining_association_rules", "lift")]},
               title="Association rules", order=2),
        visual("ms_s_task", "slicer", 880, 64, 180, 200,
               roles={"Values": [col("mining_model_metrics", "task")]},
               title="Task", order=3),
        card("ms_c1", "Rules", 880, 284, 4, w=180),
        card("ms_c2", "Max Lift", 880, 404, 5, w=180),
        card("ms_c3", "Segments Clustered", 1080, 284, 6, w=180),
        card("ms_c4", "Anomalies", 1080, 404, 7, w=180),
    ]
    pages.append(("mining-scorecard", "6. Mining scorecard", v))

    return pages


# ================================================================ file writing
def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def platform(item_type: str, display_name: str) -> dict:
    return {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/"
                   "gitIntegration/platformProperties/2.0.0/schema.json",
        "metadata": {"type": item_type, "displayName": display_name},
        "config": {"version": "2.0", "logicalId": gid("item", item_type, display_name)},
    }


def build_semantic_model(root: Path, data_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    model_dir = root / f"{PROJECT}.SemanticModel"
    if (model_dir / "definition").exists():
        shutil.rmtree(model_dir / "definition")

    write_json(model_dir / ".platform", platform("SemanticModel", PROJECT))
    write_json(model_dir / "definition.pbism", {"version": "4.2", "settings": {}})

    write_text(model_dir / "definition" / "database.tmdl",
               "database\n\tcompatibilityLevel: 1567\n")

    # DataFolder is the single knob a new machine has to turn.
    write_text(
        model_dir / "definition" / "expressions.tmdl",
        "/// Absolute path to the folder written by `python -m src.export_bi`.\n"
        "/// Change this one value after cloning, then Refresh.\n"
        "expression DataFolder = "
        f'"{data_dir.as_posix()}/" meta [IsParameterQuery = true, '
        "Type = \"Text\", IsParameterQueryRequired = true]\n"
        "\tlineageTag: " + gid("expression", "DataFolder") + "\n"
        "\n"
        "\tannotation PBI_NavigationStepName = Navigation\n"
        "\n"
        "\tannotation PBI_ResultType = Text\n"
    )

    ordered = list(tables)
    for table, frame in tables.items():
        write_text(model_dir / "definition" / "tables" / f"{table}.tmdl",
                   table_tmdl(table, frame))
    write_text(model_dir / "definition" / "tables" / f"{MEASURE_TABLE}.tmdl",
               measures_tmdl())

    write_text(model_dir / "definition" / "relationships.tmdl",
               relationships_tmdl(set(tables)))

    refs = "\n".join(f"ref table {quote(t)}" for t in ordered + [MEASURE_TABLE])
    write_text(
        model_dir / "definition" / "model.tmdl",
        "model Model\n"
        "\tculture: en-GB\n"
        "\tdefaultPowerBIDataSourceVersion: powerBI_V3\n"
        "\tdiscourageImplicitMeasures\n"
        "\tsourceQueryCulture: en-GB\n"
        "\tdataAccessOptions\n"
        "\t\tlegacyRedirects\n"
        "\t\treturnErrorValuesAsNull\n"
        "\n"
        "annotation PBI_QueryOrder = " + json.dumps(ordered) + "\n"
        "\n"
        "ref expression DataFolder\n"
        "\n" + refs + "\n"
    )


def build_report(root: Path) -> None:
    report_dir = root / f"{PROJECT}.Report"
    if (report_dir / "definition").exists():
        shutil.rmtree(report_dir / "definition")

    write_json(report_dir / ".platform", platform("Report", PROJECT))
    write_json(report_dir / "definition.pbir", {
        "$schema": "https://developer.microsoft.com/json-schemas/fabric/item/"
                   "report/definitionProperties/1.0.0/schema.json",
        "version": "4.0",
        "datasetReference": {"byPath": {"path": f"../{PROJECT}.SemanticModel"}},
    })

    definition = report_dir / "definition"
    write_json(definition / "report.json", {
        "$schema": f"{SCHEMA}/report/1.0.0/schema.json",
        "themeCollection": {"baseTheme": {"name": "CY24SU10",
                                          "reportVersionAtImport": "5.55",
                                          "type": "SharedResources"}},
        "layoutOptimization": "None",
        "settings": {"useStylableVisualContainerHeader": True},
    })

    pages = build_pages()
    for page_id, display_name, visuals in pages:
        page_dir = definition / "pages" / page_id
        write_json(page_dir / "page.json", {
            "$schema": f"{SCHEMA}/page/1.0.0/schema.json",
            "name": page_id,
            "displayName": display_name,
            "displayOption": "FitToPage",
            "height": PAGE_H,
            "width": PAGE_W,
        })
        for spec in visuals:
            # Folder named after the logical id, not the GUID, so a changed
            # visual shows up as a readable path in review.
            write_json(page_dir / "visuals" / spec["slug"] / "visual.json",
                       {k: v for k, v in spec.items() if k != "slug"})

    write_json(definition / "pages" / "pages.json", {
        "$schema": f"{SCHEMA}/pagesMetadata/1.0.0/schema.json",
        "pageOrder": [page_id for page_id, _, _ in pages],
        "activePageName": pages[0][0],
    })


def main(data_dir: Path | str | None = None) -> Path:
    data = Path(data_dir).resolve() if data_dir else (HERE / "data").resolve()
    csvs = sorted(p for p in data.glob("*.csv")
                  if p.stem not in {"export_manifest"})
    if not csvs:
        raise SystemExit(
            f"no exported CSVs in {data} — run `python -m src.export_bi` first")

    # Only headers and dtypes are needed, so read a sample rather than 525k rows.
    tables = {p.stem: pd.read_csv(p, nrows=500, encoding="utf-8-sig") for p in csvs}
    # Dimensions before facts before mining satellites, which is the order the
    # field list shows and the order Power Query loads in.
    order = {"dim": 0, "fac": 1, "agg": 2}
    tables = dict(sorted(tables.items(),
                         key=lambda kv: (order.get(kv[0][:3], 3), kv[0])))

    build_semantic_model(HERE, data, tables)
    build_report(HERE)

    print(f"PBIP project written to {HERE}")
    print(f"  semantic model : {len(tables)} tables + {len(MEASURES)} measures "
          f"+ {len(RELATIONSHIPS)} relationships")
    print(f"  report         : {len(build_pages())} pages")
    print(f"\nOpen {HERE / (PROJECT + '.pbip')} in Power BI Desktop and Refresh.")
    return HERE


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="exported CSV folder (default powerbi/data)")
    main(ap.parse_args().data)

-- ============================================================================
-- Traffic Incident & Congestion Data Warehouse — star schema (two fact tables
-- sharing conformed dimensions, i.e. a fact constellation / galaxy schema).
--
--   dim_time ──┐                  ┌── dim_segment
--              ├── fact_traffic ──┤
--   dim_weather┘                  └── dim_segment
--              ├── fact_incident ─┤
--   dim_incident_type ────────────┘
--
-- Grain: fact_traffic  = one row per road segment per hour.
--        fact_incident = one row per reported incident.
-- ============================================================================

DROP TABLE IF EXISTS fact_incident;
DROP TABLE IF EXISTS fact_traffic;
DROP TABLE IF EXISTS dim_time;
DROP TABLE IF EXISTS dim_segment;
DROP TABLE IF EXISTS dim_weather;
DROP TABLE IF EXISTS dim_incident_type;

-- ---------------------------------------------------------------- dimensions
CREATE TABLE dim_time (
    time_key      INTEGER PRIMARY KEY,   -- yyyymmddhh
    timestamp     TEXT    NOT NULL,
    date          TEXT    NOT NULL,
    year          INTEGER NOT NULL,
    quarter       INTEGER NOT NULL,
    month         INTEGER NOT NULL,
    month_name    TEXT    NOT NULL,
    week_of_year  INTEGER NOT NULL,
    day_of_month  INTEGER NOT NULL,
    day_of_week   INTEGER NOT NULL,
    day_name      TEXT    NOT NULL,
    hour          INTEGER NOT NULL,
    time_of_day   TEXT    NOT NULL,
    is_weekend    INTEGER NOT NULL,
    is_peak_hour  INTEGER NOT NULL
);

CREATE TABLE dim_segment (
    segment_key      INTEGER PRIMARY KEY,
    segment_id       TEXT    NOT NULL UNIQUE,
    segment_name     TEXT,
    zone             TEXT    NOT NULL,   -- roll-up: segment -> zone -> city
    road_type        TEXT    NOT NULL,
    lanes            INTEGER NOT NULL,
    length_km        REAL    NOT NULL,
    speed_limit_kmph INTEGER NOT NULL,
    signalised       INTEGER NOT NULL,
    latitude         REAL,
    longitude        REAL
);

CREATE TABLE dim_weather (
    weather_key   INTEGER PRIMARY KEY,
    weather       TEXT NOT NULL UNIQUE,
    is_adverse    INTEGER NOT NULL
);

CREATE TABLE dim_incident_type (
    incident_type_key INTEGER PRIMARY KEY,
    incident_type     TEXT NOT NULL UNIQUE,
    category          TEXT NOT NULL       -- Accident / Obstruction / Planned
);

-- --------------------------------------------------------------- fact tables
CREATE TABLE fact_traffic (
    traffic_key      INTEGER PRIMARY KEY,
    time_key         INTEGER NOT NULL REFERENCES dim_time(time_key),
    segment_key      INTEGER NOT NULL REFERENCES dim_segment(segment_key),
    weather_key      INTEGER NOT NULL REFERENCES dim_weather(weather_key),
    vehicle_count    INTEGER,
    avg_speed_kmph   REAL,
    occupancy_pct    REAL,
    travel_time_min  REAL,
    congestion_index REAL,
    congestion_level TEXT,
    speed_ratio      REAL,
    delay_index      REAL,
    flow_per_lane    REAL,
    vehicle_km       REAL,
    incident_count   INTEGER,
    is_congested     INTEGER
);

CREATE TABLE fact_incident (
    incident_key      INTEGER PRIMARY KEY,
    incident_id       TEXT    NOT NULL UNIQUE,
    time_key          INTEGER NOT NULL REFERENCES dim_time(time_key),
    segment_key       INTEGER NOT NULL REFERENCES dim_segment(segment_key),
    weather_key       INTEGER NOT NULL REFERENCES dim_weather(weather_key),
    incident_type_key INTEGER NOT NULL REFERENCES dim_incident_type(incident_type_key),
    severity          INTEGER,
    vehicles_involved INTEGER,
    lanes_blocked     INTEGER,
    casualties        INTEGER,
    duration_min      REAL,
    response_time_min REAL,
    congestion_index  REAL,
    is_major          INTEGER
);

CREATE INDEX idx_ft_time    ON fact_traffic(time_key);
CREATE INDEX idx_ft_segment ON fact_traffic(segment_key);
CREATE INDEX idx_ft_weather ON fact_traffic(weather_key);
CREATE INDEX idx_fi_time    ON fact_incident(time_key);
CREATE INDEX idx_fi_segment ON fact_incident(segment_key);
CREATE INDEX idx_fi_type    ON fact_incident(incident_type_key);

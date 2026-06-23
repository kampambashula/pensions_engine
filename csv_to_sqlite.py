"""
csv_to_sqlite.py
----------------
Production-grade CSV-to-SQLite importer for the Pension Calculation Engine.

Responsibilities:
  - Ingest pension_analysis_ready.csv
  - Create a strongly-typed SQLite schema optimised for actuarial queries
  - Insert records with type coercion and validation
  - Build indexes for all high-frequency query paths
  - Produce an import summary report

Author  : Pension Actuarial Systems — Data Engineering Layer
"""

from __future__ import annotations

import logging
import os
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "pension_engine.db"
DEFAULT_CSV = DATA_DIR / "pension_analysis_ready.csv"

LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("csv_to_sqlite")

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

DDL_MEMBERS = """
CREATE TABLE IF NOT EXISTS members (
    -- Identity
    row_id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    member_id                   TEXT        NOT NULL UNIQUE,
    employee_number             TEXT,
    nrc                         TEXT,

    -- Personal
    first_name                  TEXT,
    last_name                   TEXT,
    gender                      TEXT        CHECK(gender IN ('Male','Female','Unknown')),
    date_of_birth               DATE,

    -- Employment
    date_joined_employer        DATE,
    date_joined_scheme          DATE,
    employment_status           TEXT,
    department                  TEXT,
    sector                      TEXT,

    -- Age & Service
    current_age                 REAL,
    service_years               REAL,
    pensionable_service_years   REAL,
    years_to_retirement         REAL,
    retirement_age              REAL,
    normal_retirement_date      DATE,

    -- Salary
    basic_salary                REAL        DEFAULT 0,
    housing_allowance           REAL        DEFAULT 0,
    transport_allowance         REAL        DEFAULT 0,
    total_monthly_salary        REAL        DEFAULT 0,
    pensionable_salary          REAL        DEFAULT 0,
    annual_pensionable_salary   REAL        DEFAULT 0,

    -- Contributions (monthly)
    employee_contribution_rate  REAL        DEFAULT 0,
    employer_contribution_rate  REAL        DEFAULT 0,
    employee_monthly_contribution   REAL    DEFAULT 0,
    employer_monthly_contribution   REAL    DEFAULT 0,
    total_monthly_contribution  REAL        DEFAULT 0,

    -- Accumulated Fund
    ee_accumulated_contributions    REAL    DEFAULT 0,
    er_accumulated_contributions    REAL    DEFAULT 0,
    total_accumulated_contributions REAL    DEFAULT 0,

    -- Actuarial Assumptions
    investment_return_assumption REAL       DEFAULT 0.08,
    salary_growth_assumption    REAL        DEFAULT 0.05,
    inflation_assumption        REAL        DEFAULT 0.065,
    accrual_rate                REAL        DEFAULT 0.02,
    withdrawal_rate             REAL        DEFAULT 0.03,
    disability_rate             REAL        DEFAULT 0.005,
    spouse_percentage           REAL        DEFAULT 0.60,
    mortality_table             TEXT,

    -- Benefit Projection
    benefit_formula_type        TEXT,
    pension_type                TEXT,
    projected_final_salary      REAL        DEFAULT 0,
    projected_pension_benefit   REAL        DEFAULT 0,

    -- Valuation Meta
    last_valuation_date         DATE,
    status                      TEXT,

    -- Computed at load time
    loaded_at                   DATETIME    DEFAULT CURRENT_TIMESTAMP
);
"""

DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_members_member_id          ON members(member_id);",
    "CREATE INDEX IF NOT EXISTS idx_members_status              ON members(status);",
    "CREATE INDEX IF NOT EXISTS idx_members_department          ON members(department);",
    "CREATE INDEX IF NOT EXISTS idx_members_employment_status   ON members(employment_status);",
    "CREATE INDEX IF NOT EXISTS idx_members_current_age         ON members(current_age);",
    "CREATE INDEX IF NOT EXISTS idx_members_years_to_retirement ON members(years_to_retirement);",
    "CREATE INDEX IF NOT EXISTS idx_members_gender              ON members(gender);",
    "CREATE INDEX IF NOT EXISTS idx_members_pension_type        ON members(pension_type);",
]

DDL_SCENARIO_RUNS = """
CREATE TABLE IF NOT EXISTS scenario_runs (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_name            TEXT        NOT NULL,
    run_at              DATETIME    DEFAULT CURRENT_TIMESTAMP,
    salary_growth       REAL,
    inflation           REAL,
    investment_return   REAL,
    retirement_age_delta REAL       DEFAULT 0,
    total_liability     REAL,
    funding_ratio       REAL,
    notes               TEXT
);
"""

DDL_IMPORT_LOG = """
CREATE TABLE IF NOT EXISTS import_log (
    log_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    imported_at     DATETIME    DEFAULT CURRENT_TIMESTAMP,
    source_file     TEXT,
    rows_total      INTEGER,
    rows_inserted   INTEGER,
    rows_skipped    INTEGER,
    duration_secs   REAL,
    notes           TEXT
);
"""

# ---------------------------------------------------------------------------
# Column Mapping  (CSV column → DB column)
# ---------------------------------------------------------------------------

# Maps the incoming CSV header (after pandas normalisation) to the DB column.
# If a CSV column doesn't appear here it is silently ignored.
COLUMN_MAP: dict[str, str] = {
    "member_id":                        "member_id",
    "employee_number":                  "employee_number",
    "nrc":                              "nrc",
    "reference_number":                 "member_id",   # contributions.csv alias
    "first_name":                       "first_name",
    "forename":                         "first_name",
    "forenames":                        "first_name",
    "last_name":                        "last_name",
    "surname":                          "last_name",
    "gender":                           "gender",
    "date_of_birth":                    "date_of_birth",
    "dob":                              "date_of_birth",
    "date_joined_employer":             "date_joined_employer",
    "doe":                              "date_joined_employer",
    "date_joined_scheme":               "date_joined_scheme",
    "employment_status":                "employment_status",
    "department":                       "department",
    "sector":                           "sector",
    "current_age":                      "current_age",
    "service_years":                    "service_years",
    "pensionable_service_years":        "pensionable_service_years",
    "years_to_retirement":              "years_to_retirement",
    "retirement_age":                   "retirement_age",
    "normal_retirement_date":           "normal_retirement_date",
    "basic_salary":                     "basic_salary",
    "housing_allowance":                "housing_allowance",
    "transport_allowance":              "transport_allowance",
    "total_monthly_salary":             "total_monthly_salary",
    "pensionable_salary":               "pensionable_salary",
    "annual_pensionable_salary":        "annual_pensionable_salary",
    "employee_contribution_rate":       "employee_contribution_rate",
    "employer_contribution_rate":       "employer_contribution_rate",
    "employee_monthly_contribution":    "employee_monthly_contribution",
    "employer_monthly_contribution":    "employer_monthly_contribution",
    "total_monthly_contribution":       "total_monthly_contribution",
    "ee_accumulated_contributions":     "ee_accumulated_contributions",
    "er_accumulated_contributions":     "er_accumulated_contributions",
    "total_accumulated_contributions":  "total_accumulated_contributions",
    "investment_return_assumption":     "investment_return_assumption",
    "salary_growth_assumption":         "salary_growth_assumption",
    "inflation_assumption":             "inflation_assumption",
    "accrual_rate":                     "accrual_rate",
    "withdrawal_rate":                  "withdrawal_rate",
    "disability_rate":                  "disability_rate",
    "spouse_percentage":                "spouse_percentage",
    "mortality_table":                  "mortality_table",
    "benefit_formula_type":             "benefit_formula_type",
    "pension_type":                     "pension_type",
    "projected_final_salary":           "projected_final_salary",
    "projected_pension_benefit":        "projected_pension_benefit",
    "last_valuation_date":              "last_valuation_date",
    "status":                           "status",
    # contributions.csv legacy columns
    "annual_salary":                    "annual_pensionable_salary",
    "member_contributions_2023":        "employee_monthly_contribution",
    "employer_contributions":           "employer_monthly_contribution",
    "percent_contribution":             "employee_contribution_rate",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Lowercase + snake_case column names."""
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^\w]", "_", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
    )
    return df


def _normalise_gender(series: pd.Series) -> pd.Series:
    mapping = {
        "m": "Male", "male": "Male",
        "f": "Female", "female": "Female",
    }
    return series.str.strip().str.lower().map(mapping).fillna("Unknown")


def _parse_numeric(series: pd.Series) -> pd.Series:
    """Strip currency symbols / commas and coerce to float."""
    return (
        series.astype(str)
        .str.replace(r"[^\d.\-]", "", regex=True)
        .replace("", "0")
        .astype(float)
    )


def _parse_date(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce").dt.strftime("%Y-%m-%d")


NUMERIC_COLS = {
    "current_age", "service_years", "pensionable_service_years",
    "years_to_retirement", "retirement_age",
    "basic_salary", "housing_allowance", "transport_allowance",
    "total_monthly_salary", "pensionable_salary", "annual_pensionable_salary",
    "employee_contribution_rate", "employer_contribution_rate",
    "employee_monthly_contribution", "employer_monthly_contribution",
    "total_monthly_contribution",
    "ee_accumulated_contributions", "er_accumulated_contributions",
    "total_accumulated_contributions",
    "investment_return_assumption", "salary_growth_assumption",
    "inflation_assumption", "accrual_rate",
    "withdrawal_rate", "disability_rate", "spouse_percentage",
    "projected_final_salary", "projected_pension_benefit",
}

DATE_COLS = {
    "date_of_birth", "date_joined_employer", "date_joined_scheme",
    "normal_retirement_date", "last_valuation_date",
}


# ---------------------------------------------------------------------------
# Core import logic
# ---------------------------------------------------------------------------

def initialise_database(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()
    cur.executescript(
        DDL_MEMBERS + DDL_SCENARIO_RUNS + DDL_IMPORT_LOG
    )
    for idx in DDL_INDEXES:
        cur.execute(idx)
    conn.commit()
    log.info("Database schema initialised.")


def load_csv(csv_path: Path) -> pd.DataFrame:
    log.info("Loading CSV: %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False, encoding="utf-8-sig")
    log.info("Raw shape: %s rows × %s cols", *df.shape)
    df = _normalise_headers(df)
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map CSV columns → DB columns, apply type coercions,
    enforce gender normalisation, return a clean DataFrame.
    """
    # Rename using column map
    rename_map = {c: COLUMN_MAP[c] for c in df.columns if c in COLUMN_MAP}
    df = df.rename(columns=rename_map)

    # Keep only DB columns
    db_cols = set(COLUMN_MAP.values())
    df = df[[c for c in df.columns if c in db_cols]].copy()

    # Gender
    if "gender" in df.columns:
        df["gender"] = _normalise_gender(df["gender"])

    # Numeric
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = _parse_numeric(df[col])

    # Dates
    for col in DATE_COLS:
        if col in df.columns:
            df[col] = _parse_date(df[col])

    # Ensure member_id is string
    if "member_id" in df.columns:
        df["member_id"] = df["member_id"].astype(str).str.strip()

    # Drop rows without a member_id
    before = len(df)
    df = df.dropna(subset=["member_id"])
    df = df[df["member_id"].str.strip() != ""]
    after = len(df)
    if before != after:
        log.warning("Dropped %d rows missing member_id.", before - after)

    # Deduplicate on member_id (keep last)
    df = df.drop_duplicates(subset=["member_id"], keep="last")
    log.info("Transformed shape: %d rows", len(df))
    return df


def insert_members(conn: sqlite3.Connection, df: pd.DataFrame) -> tuple[int, int]:
    """
    Bulk insert with ON CONFLICT REPLACE for idempotent reruns.
    Returns (inserted, skipped).
    """
    db_cols = [col for col in df.columns]
    placeholders = ", ".join(["?"] * len(db_cols))
    col_names = ", ".join(db_cols)

    sql = (
        f"INSERT OR REPLACE INTO members ({col_names}) "
        f"VALUES ({placeholders})"
    )

    records = [
        tuple(None if pd.isna(v) else v for v in row)
        for row in df.itertuples(index=False, name=None)
    ]

    cur = conn.cursor()
    cur.executemany(sql, records)
    conn.commit()
    inserted = cur.rowcount if cur.rowcount >= 0 else len(records)
    return len(records), 0


def log_import(
    conn: sqlite3.Connection,
    source_file: str,
    rows_total: int,
    rows_inserted: int,
    rows_skipped: int,
    duration: float,
) -> None:
    conn.execute(
        "INSERT INTO import_log "
        "(source_file, rows_total, rows_inserted, rows_skipped, duration_secs) "
        "VALUES (?,?,?,?,?)",
        (source_file, rows_total, rows_inserted, rows_skipped, round(duration, 3)),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_import(csv_path: Path | None = None, db_path: Path | None = None) -> dict[str, Any]:
    csv_path = csv_path or DEFAULT_CSV
    db_path  = db_path  or DB_PATH

    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        sys.exit(1)

    t0 = time.perf_counter()
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA cache_size=-64000;")   # 64 MB cache

    initialise_database(conn)
    df_raw = load_csv(csv_path)
    df     = transform(df_raw)

    rows_total = len(df)
    inserted, skipped = insert_members(conn, df)
    duration = time.perf_counter() - t0

    log_import(conn, str(csv_path), rows_total, inserted, skipped, duration)
    conn.close()

    summary = {
        "db_path":       str(db_path),
        "source_file":   str(csv_path),
        "rows_total":    rows_total,
        "rows_inserted": inserted,
        "rows_skipped":  skipped,
        "duration_secs": round(duration, 2),
    }
    log.info("Import complete: %s", summary)
    return summary


if __name__ == "__main__":
    csv_override = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    result = run_import(csv_override)
    print("\n=== Import Summary ===")
    for k, v in result.items():
        print(f"  {k:<20} {v}")

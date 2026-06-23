"""
calculation_engine.py
---------------------
Pension Actuarial Calculation Engine.

Implements:
  - Projected Unit Credit (PUC) benefit projections
  - Defined Benefit (DB) and Defined Contribution (DC) valuation logic
  - Funding ratio analysis
  - Portfolio-level liability estimation
  - Contribution adequacy assessment
  - Retirement exposure profiling
  - Dynamic scenario testing
  - High-risk member detection
  - Actuarial summary reporting

All monetary values are in the scheme's reporting currency.
All rates are expressed as decimals (e.g. 0.08 = 8%).

Author  : Pension Actuarial Systems — Actuarial Model Layer
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import logging

import numpy as np
import pandas as pd

from actuarial_tables import (
    annuity_factor as _actuarial_annuity_factor,
    get_mortality_qx,
    get_ill_health_qx,
)

log = logging.getLogger("calculation_engine")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH  = BASE_DIR / "pension_engine.db"

# ---------------------------------------------------------------------------
# Actuarial constants & defaults
# ---------------------------------------------------------------------------

DEFAULT_ASSUMPTIONS = {
    "salary_growth":        0.05,   # 5 % p.a.
    "inflation":            0.065,  # 6.5 % p.a.
    "investment_return":    0.08,   # 8 % p.a.
    "accrual_rate":         0.02,   # 1/50th accrual
    "withdrawal_rate":      0.03,
    "disability_rate":      0.005,
    "spouse_percentage":    0.60,
    "mortality_improvement":0.01,
}

NEAR_RETIREMENT_YEARS  = 5     # members within 5 years of NRD
HIGH_RISK_CONTRIB_MULT = 0.20  # accumulated < 20 % of projected benefit → high risk
HIGH_CONCENTRATION_PCT = 0.02  # single member > 2 % of total liability → concentration risk


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------

def get_connection(db_path: Path | None = None) -> sqlite3.Connection:
    path = db_path or DB_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Database not found at {path}. Run csv_to_sqlite.py first."
        )
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON;")
    return conn


def load_members(db_path: Path | None = None) -> pd.DataFrame:
    """Load full members table into a DataFrame."""
    with get_connection(db_path) as conn:
        df = pd.read_sql_query("SELECT * FROM members", conn)
    _coerce_numerics(df)
    log.info("Loaded %d member records.", len(df))
    return df


def _coerce_numerics(df: pd.DataFrame) -> None:
    numeric_cols = [
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
        "projected_final_salary", "projected_pension_benefit",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)


# ---------------------------------------------------------------------------
# Benefit projection functions
# ---------------------------------------------------------------------------

def project_final_salary(
    current_salary: float,
    years_to_retirement: float,
    salary_growth: float,
) -> float:
    """
    Project final pensionable salary using compound salary growth.
    S_n = S_0 × (1 + g)^n
    """
    if years_to_retirement <= 0:
        return current_salary
    return current_salary * ((1 + salary_growth) ** years_to_retirement)


def projected_pension_db(
    projected_final_salary: float,
    total_service_at_retirement: float,
    accrual_rate: float,
) -> float:
    """
    Defined Benefit formula (Career Average / Final Salary):
    P = accrual_rate × total_service × projected_final_salary
    Represents annual pension at retirement.
    """
    return accrual_rate * total_service_at_retirement * projected_final_salary


def projected_pension_dc(
    accumulated_fund: float,
    years_to_retirement: float,
    monthly_contribution: float,
    investment_return: float,
    annuity_factor: float = 12.0,
) -> float:
    """
    Defined Contribution projection.
    FV = accumulated_fund × (1+r)^n + monthly_contribution × 12 × [(1+r)^n - 1] / r
    Annual pension = FV / annuity_factor (simplified)
    """
    if years_to_retirement <= 0:
        return accumulated_fund / annuity_factor if annuity_factor else 0
    r = investment_return
    n = years_to_retirement
    fv_existing = accumulated_fund * ((1 + r) ** n)
    if r > 0:
        fv_contributions = (monthly_contribution * 12) * (((1 + r) ** n) - 1) / r
    else:
        fv_contributions = monthly_contribution * 12 * n
    total_fund = fv_existing + fv_contributions
    return total_fund / annuity_factor if annuity_factor else total_fund


def present_value_liability(
    annual_pension: float,
    years_to_retirement: float,
    investment_return: float,
    annuity_factor: float = 12.0,
    retirement_age: float = 0.0,
    gender: str = "Unknown",
) -> float:
    """
    Discounted Present Value of projected pension liability.

    When retirement_age and gender are supplied, ä_x is computed from the
    a(55) immediate annuitant mortality table at the expected retirement age.
    Otherwise falls back to the supplied annuity_factor parameter.

      PV = annual_pension × ä_{ret_age}(r) / (1 + r)^n
    """
    if investment_return <= 0 or years_to_retirement < 0:
        return annual_pension * annuity_factor

    if retirement_age > 0 and gender in ("Male", "Female"):
        try:
            ax = _actuarial_annuity_factor(retirement_age, gender, investment_return)
        except Exception:
            ax = annuity_factor
    else:
        ax = annuity_factor

    discount_factor = (1 + investment_return) ** years_to_retirement
    return (annual_pension * ax) / discount_factor


# ---------------------------------------------------------------------------
# Member-level enrichment
# ---------------------------------------------------------------------------

def enrich_members(
    df: pd.DataFrame,
    assumptions: dict | None = None,
) -> pd.DataFrame:
    """
    Compute derived actuarial fields on the full member DataFrame.
    Returns the DataFrame with additional computed columns.
    """
    a = {**DEFAULT_ASSUMPTIONS, **(assumptions or {})}
    df = df.copy()

    # Use per-member assumptions where available, else global default
    sg = df["salary_growth_assumption"].where(df["salary_growth_assumption"] > 0, a["salary_growth"])
    ir = df["investment_return_assumption"].where(df["investment_return_assumption"] > 0, a["investment_return"])
    ar = df["accrual_rate"].where(df["accrual_rate"] > 0, a["accrual_rate"])

    ytr = df["years_to_retirement"].clip(lower=0)

    # Projected final salary (use existing if populated, else compute)
    mask_no_proj = df["projected_final_salary"] <= 0
    df.loc[mask_no_proj, "projected_final_salary"] = df.loc[mask_no_proj].apply(
        lambda r: project_final_salary(
            r["annual_pensionable_salary"],
            r["years_to_retirement"],
            sg.loc[r.name],
        ),
        axis=1,
    )

    # Total service at retirement
    df["service_at_retirement"] = (
        df["pensionable_service_years"].clip(lower=0) + ytr
    )

    # DB projected benefit
    df["calc_pension_db"] = projected_pension_db(
        df["projected_final_salary"],
        df["service_at_retirement"],
        ar,
    )

    # DC projected benefit
    df["calc_pension_dc"] = df.apply(
        lambda r: projected_pension_dc(
            r["total_accumulated_contributions"],
            r["years_to_retirement"],
            r["total_monthly_contribution"],
            ir.loc[r.name],
        ),
        axis=1,
    )

    # Unified projected benefit (use formula type if available)
    def _pick_benefit(row: pd.Series) -> float:
        ft = str(row.get("benefit_formula_type", "")).strip().upper()
        if ft in ("DC", "DEFINED CONTRIBUTION"):
            return row["calc_pension_dc"]
        return row["calc_pension_db"]   # Default: DB

    df["unified_projected_benefit"] = df.apply(_pick_benefit, axis=1)

    # Present value of liability — uses a(55) mortality-derived annuity factor
    df["pv_liability"] = df.apply(
        lambda r: present_value_liability(
            r["unified_projected_benefit"],
            r["years_to_retirement"],
            ir.loc[r.name],
            annuity_factor=12.0,
            retirement_age=float(r.get("retirement_age", 0) or 0),
            gender=str(r.get("gender", "Unknown") or "Unknown"),
        ),
        axis=1,
    )

    # Ill-health inception rate (CMI WP50 ACMNL04/ACFNL04) per member
    df["ill_health_qx"] = df.apply(
        lambda r: get_ill_health_qx(
            float(r.get("current_age", 40) or 40),
            str(r.get("gender", "Unknown") or "Unknown"),
            duration=5,
        ),
        axis=1,
    )

    # Mortality rate from a(55) table per member
    df["mortality_qx"] = df.apply(
        lambda r: get_mortality_qx(
            float(r.get("current_age", 40) or 40),
            str(r.get("gender", "Unknown") or "Unknown"),
            duration=1,
        ),
        axis=1,
    )

    # Contribution adequacy ratio
    # = total accumulated / PV liability  (>1 means fully funded at member level)
    df["contribution_adequacy_ratio"] = np.where(
        df["pv_liability"] > 0,
        df["total_accumulated_contributions"] / df["pv_liability"],
        np.nan,
    )

    # Annual contribution (annualised monthly)
    df["annual_contribution"] = df["total_monthly_contribution"] * 12

    # Near retirement flag
    df["near_retirement"] = ytr <= NEAR_RETIREMENT_YEARS

    # Age band
    df["age_band"] = pd.cut(
        df["current_age"],
        bins=[0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 999],
        labels=["<25","25-29","30-34","35-39","40-44","45-49","50-54","55-59","60-64","65+"],
        right=False,
    )

    return df


# ---------------------------------------------------------------------------
# Portfolio-level analytics
# ---------------------------------------------------------------------------

@dataclass
class PortfolioMetrics:
    total_members:              int
    active_members:             int
    deferred_members:           int
    retired_members:            int

    total_annual_payroll:       float
    total_monthly_contributions:float
    total_annual_contributions: float
    total_ee_accumulated:       float
    total_er_accumulated:       float
    total_accumulated:          float

    total_pv_liability:         float
    total_assets:               float      # = total_accumulated as proxy
    funding_ratio:              float

    avg_age:                    float
    avg_service:                float
    avg_projected_benefit:      float

    near_retirement_count:      int
    high_risk_count:            int

    liability_top10_pct:        float
    concentration_risk_members: int


def compute_portfolio_metrics(df: pd.DataFrame) -> PortfolioMetrics:
    status_lower = df["employment_status"].str.lower().fillna("")

    active   = df[status_lower.isin(["active", "employed"])].shape[0]
    deferred = df[status_lower.isin(["deferred", "preserved"])].shape[0]
    retired  = df[status_lower.isin(["retired", "pensioner"])].shape[0]

    total_pv = df["pv_liability"].sum()
    total_acc = df["total_accumulated_contributions"].sum()
    funding   = total_acc / total_pv if total_pv > 0 else np.nan

    near_ret  = df["near_retirement"].sum()
    high_risk = identify_high_risk(df).shape[0]

    # Top-10 member liability as % of total
    top10_liab = df["pv_liability"].nlargest(10).sum()
    top10_pct  = top10_liab / total_pv if total_pv > 0 else 0

    conc_threshold = total_pv * HIGH_CONCENTRATION_PCT
    conc_count = (df["pv_liability"] > conc_threshold).sum()

    return PortfolioMetrics(
        total_members               = len(df),
        active_members              = active,
        deferred_members            = deferred,
        retired_members             = retired,
        total_annual_payroll        = df["annual_pensionable_salary"].sum(),
        total_monthly_contributions = df["total_monthly_contribution"].sum(),
        total_annual_contributions  = df["annual_contribution"].sum(),
        total_ee_accumulated        = df["ee_accumulated_contributions"].sum(),
        total_er_accumulated        = df["er_accumulated_contributions"].sum(),
        total_accumulated           = total_acc,
        total_pv_liability          = total_pv,
        total_assets                = total_acc,
        funding_ratio               = funding,
        avg_age                     = df["current_age"].mean(),
        avg_service                 = df["service_years"].mean(),
        avg_projected_benefit       = df["unified_projected_benefit"].mean(),
        near_retirement_count       = int(near_ret),
        high_risk_count             = int(high_risk),
        liability_top10_pct         = top10_pct,
        concentration_risk_members  = int(conc_count),
    )


# ---------------------------------------------------------------------------
# Risk identification
# ---------------------------------------------------------------------------

def identify_high_risk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return members meeting one or more high-risk criteria:
      1. Contribution adequacy ratio < HIGH_RISK_CONTRIB_MULT
      2. Near retirement with adequacy ratio < 0.50
      3. Zero or negative accumulated contributions
      4. Liability concentration > HIGH_CONCENTRATION_PCT of total
    """
    total_pv = df["pv_liability"].sum()

    mask = (
        (df["contribution_adequacy_ratio"] < HIGH_RISK_CONTRIB_MULT)
        | (df["near_retirement"] & (df["contribution_adequacy_ratio"] < 0.50))
        | (df["total_accumulated_contributions"] <= 0)
        | (df["pv_liability"] > total_pv * HIGH_CONCENTRATION_PCT)
    )

    risk_df = df[mask].copy()

    def _reason(row: pd.Series) -> str:
        reasons = []
        if row["total_accumulated_contributions"] <= 0:
            reasons.append("Zero accumulated contributions")
        if row["contribution_adequacy_ratio"] < HIGH_RISK_CONTRIB_MULT:
            reasons.append("Low adequacy ratio")
        if row["near_retirement"] and row["contribution_adequacy_ratio"] < 0.50:
            reasons.append("Near retirement / underfunded")
        if row["pv_liability"] > total_pv * HIGH_CONCENTRATION_PCT:
            reasons.append("Liability concentration")
        return "; ".join(reasons) if reasons else "Other"

    risk_df["risk_reason"] = risk_df.apply(_reason, axis=1)
    return risk_df.sort_values("pv_liability", ascending=False)


# ---------------------------------------------------------------------------
# Scenario analysis
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    scenario_name:           str
    salary_growth:           float
    inflation:               float
    investment_return:       float
    retirement_age_delta:    float
    total_pv_liability:      float
    funding_ratio:           float
    avg_projected_benefit:   float
    near_retirement_count:   int
    delta_liability_pct:     float = 0.0
    delta_funding_pct:       float = 0.0


def run_scenario(
    df_base: pd.DataFrame,
    scenario_name: str,
    salary_growth:       float | None = None,
    inflation:           float | None = None,
    investment_return:   float | None = None,
    retirement_age_delta:float = 0.0,
    base_metrics: PortfolioMetrics | None = None,
) -> ScenarioResult:
    """
    Apply a scenario to the enriched member DataFrame and return
    updated portfolio-level metrics. Does NOT mutate df_base.
    """
    df = df_base.copy()

    overrides = {}
    if salary_growth     is not None: overrides["salary_growth"]     = salary_growth
    if inflation         is not None: overrides["inflation"]          = inflation
    if investment_return is not None: overrides["investment_return"]  = investment_return

    # Force per-member assumption columns to 0 so the engine uses global overrides
    if salary_growth is not None:
        df["salary_growth_assumption"] = salary_growth
    if investment_return is not None:
        df["investment_return_assumption"] = investment_return
    if inflation is not None:
        df["inflation_assumption"] = inflation

    # Force recalculation of projected_final_salary under new assumptions
    df["projected_final_salary"] = 0.0

    # Shift retirement age
    if retirement_age_delta != 0:
        df["years_to_retirement"] = (df["years_to_retirement"] + retirement_age_delta).clip(lower=0)

    df_scenario = enrich_members(df, assumptions=overrides)
    pm = compute_portfolio_metrics(df_scenario)

    delta_liab  = 0.0
    delta_fund  = 0.0
    if base_metrics:
        if base_metrics.total_pv_liability > 0:
            delta_liab = (pm.total_pv_liability - base_metrics.total_pv_liability) / base_metrics.total_pv_liability
        if base_metrics.funding_ratio and base_metrics.funding_ratio > 0:
            delta_fund = (pm.funding_ratio - base_metrics.funding_ratio) / base_metrics.funding_ratio

    return ScenarioResult(
        scenario_name        = scenario_name,
        salary_growth        = salary_growth or overrides.get("salary_growth", DEFAULT_ASSUMPTIONS["salary_growth"]),
        inflation            = inflation     or overrides.get("inflation",     DEFAULT_ASSUMPTIONS["inflation"]),
        investment_return    = investment_return or overrides.get("investment_return", DEFAULT_ASSUMPTIONS["investment_return"]),
        retirement_age_delta = retirement_age_delta,
        total_pv_liability   = pm.total_pv_liability,
        funding_ratio        = pm.funding_ratio,
        avg_projected_benefit= pm.avg_projected_benefit,
        near_retirement_count= pm.near_retirement_count,
        delta_liability_pct  = delta_liab,
        delta_funding_pct    = delta_fund,
    )


def run_standard_scenarios(
    df_enriched: pd.DataFrame,
    base_metrics: PortfolioMetrics,
) -> list[ScenarioResult]:
    """
    Run a standard suite of actuarial scenarios against the base position.
    """
    a = DEFAULT_ASSUMPTIONS
    scenarios = [
        dict(name="Base Case",             sg=a["salary_growth"],   ir=a["investment_return"],   inf=a["inflation"],   rad=0),
        dict(name="High Inflation (+2%)",   sg=a["salary_growth"]+.02, ir=a["investment_return"],   inf=a["inflation"]+.02, rad=0),
        dict(name="Low Returns (-2%)",      sg=a["salary_growth"],   ir=a["investment_return"]-.02, inf=a["inflation"],   rad=0),
        dict(name="High Salary Growth (+3%)",sg=a["salary_growth"]+.03,ir=a["investment_return"],  inf=a["inflation"],   rad=0),
        dict(name="Early Retirement (-2yr)",sg=a["salary_growth"],   ir=a["investment_return"],   inf=a["inflation"],   rad=-2),
        dict(name="Late Retirement (+2yr)", sg=a["salary_growth"],   ir=a["investment_return"],   inf=a["inflation"],   rad=+2),
        dict(name="Stress Test",            sg=a["salary_growth"]+.04,ir=a["investment_return"]-.03,inf=a["inflation"]+.03,rad=-2),
    ]
    results = []
    for s in scenarios:
        r = run_scenario(
            df_enriched,
            scenario_name        = s["name"],
            salary_growth        = s["sg"],
            inflation            = s["inf"],
            investment_return    = s["ir"],
            retirement_age_delta = s["rad"],
            base_metrics         = base_metrics,
        )
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Aggregation helpers for dashboard
# ---------------------------------------------------------------------------

def age_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("age_band", observed=True)
        .agg(member_count=("member_id", "count"),
             total_liability=("pv_liability", "sum"),
             avg_salary=("annual_pensionable_salary", "mean"))
        .reset_index()
    )


def department_liability(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("department")
        .agg(member_count=("member_id", "count"),
             total_liability=("pv_liability", "sum"),
             total_contributions=("total_accumulated_contributions", "sum"),
             avg_benefit=("unified_projected_benefit", "mean"))
        .reset_index()
        .sort_values("total_liability", ascending=False)
    )


def retirement_timeline(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    df2["retirement_year"] = (
        pd.Timestamp.now().year + df2["years_to_retirement"].clip(lower=0).round()
    ).astype(int)
    return (
        df2.groupby("retirement_year")
        .agg(member_count=("member_id", "count"),
             total_liability=("pv_liability", "sum"))
        .reset_index()
        .sort_values("retirement_year")
    )


def contribution_adequacy_bands(df: pd.DataFrame) -> pd.DataFrame:
    bands = pd.cut(
        df["contribution_adequacy_ratio"].fillna(0).clip(upper=3),
        bins=[0, 0.20, 0.50, 0.80, 1.0, 1.5, 999],
        labels=["<20%","20-50%","50-80%","80-100%","100-150%","150%+"],
        right=False,
    )
    return (
        bands.value_counts(sort=False)
        .rename_axis("adequacy_band")
        .reset_index(name="member_count")
    )


def top_liability_members(df: pd.DataFrame, n: int = 20) -> pd.DataFrame:
    cols = [
        "member_id", "first_name", "last_name", "department",
        "current_age", "years_to_retirement",
        "annual_pensionable_salary", "total_accumulated_contributions",
        "pv_liability", "unified_projected_benefit",
        "contribution_adequacy_ratio",
    ]
    available = [c for c in cols if c in df.columns]
    return df[available].nlargest(n, "pv_liability")


def gender_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("gender")
        .agg(count=("member_id", "count"),
             avg_liability=("pv_liability", "mean"),
             total_liability=("pv_liability", "sum"))
        .reset_index()
    )


def funding_by_department(df: pd.DataFrame) -> pd.DataFrame:
    g = department_liability(df)
    dept_acc = (
        df.groupby("department")["total_accumulated_contributions"]
        .sum()
        .rename("total_assets")
        .reset_index()
    )
    merged = g.merge(dept_acc, on="department")
    merged["funding_ratio"] = np.where(
        merged["total_liability"] > 0,
        merged["total_assets"] / merged["total_liability"],
        np.nan,
    )
    return merged.sort_values("funding_ratio")


# ---------------------------------------------------------------------------
# Main API surface
# ---------------------------------------------------------------------------

class PensionEngine:
    """
    High-level API for the pension calculation engine.
    Typical usage:

        engine = PensionEngine()
        engine.load()
        metrics = engine.portfolio_metrics
        risk    = engine.high_risk_members
        scenarios = engine.run_scenarios()
    """

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
        self._df_raw: pd.DataFrame | None = None
        self._df:     pd.DataFrame | None = None
        self._metrics: PortfolioMetrics | None = None

    def load(self, assumptions: dict | None = None) -> "PensionEngine":
        self._df_raw = load_members(self.db_path)
        self._df     = enrich_members(self._df_raw, assumptions)
        self._metrics = compute_portfolio_metrics(self._df)
        log.info("Engine loaded — %d members.", len(self._df))
        return self

    @property
    def members(self) -> pd.DataFrame:
        self._ensure_loaded()
        return self._df

    @property
    def portfolio_metrics(self) -> PortfolioMetrics:
        self._ensure_loaded()
        return self._metrics

    @property
    def high_risk_members(self) -> pd.DataFrame:
        self._ensure_loaded()
        return identify_high_risk(self._df)

    def run_scenarios(self) -> list[ScenarioResult]:
        self._ensure_loaded()
        return run_standard_scenarios(self._df, self._metrics)

    def custom_scenario(
        self,
        name: str,
        salary_growth: float | None = None,
        inflation: float | None = None,
        investment_return: float | None = None,
        retirement_age_delta: float = 0.0,
    ) -> ScenarioResult:
        self._ensure_loaded()
        return run_scenario(
            self._df, name,
            salary_growth=salary_growth,
            inflation=inflation,
            investment_return=investment_return,
            retirement_age_delta=retirement_age_delta,
            base_metrics=self._metrics,
        )

    def get_age_distribution(self) -> pd.DataFrame:
        self._ensure_loaded()
        return age_distribution(self._df)

    def get_department_liability(self) -> pd.DataFrame:
        self._ensure_loaded()
        return department_liability(self._df)

    def get_retirement_timeline(self) -> pd.DataFrame:
        self._ensure_loaded()
        return retirement_timeline(self._df)

    def get_contribution_adequacy_bands(self) -> pd.DataFrame:
        self._ensure_loaded()
        return contribution_adequacy_bands(self._df)

    def get_top_liability_members(self, n: int = 20) -> pd.DataFrame:
        self._ensure_loaded()
        return top_liability_members(self._df, n)

    def get_gender_distribution(self) -> pd.DataFrame:
        self._ensure_loaded()
        return gender_distribution(self._df)

    def get_funding_by_department(self) -> pd.DataFrame:
        self._ensure_loaded()
        return funding_by_department(self._df)

    def _ensure_loaded(self) -> None:
        if self._df is None:
            self.load()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    engine = PensionEngine()
    engine.load()
    pm = engine.portfolio_metrics

    print("\n=== Portfolio Summary ===")
    print(f"  Total members          : {pm.total_members:,}")
    print(f"  Active                 : {pm.active_members:,}")
    print(f"  Total accumulated fund : {pm.total_accumulated:,.0f}")
    print(f"  Total PV Liability     : {pm.total_pv_liability:,.0f}")
    print(f"  Funding Ratio          : {pm.funding_ratio:.1%}")
    print(f"  Near Retirement        : {pm.near_retirement_count:,}")
    print(f"  High-Risk Members      : {pm.high_risk_count:,}")

    print("\n=== Standard Scenarios ===")
    for s in engine.run_scenarios():
        print(
            f"  {s.scenario_name:<30} "
            f"Liability: {s.total_pv_liability:>15,.0f}  "
            f"Funding: {s.funding_ratio:.1%}  "
            f"ΔPVL: {s.delta_liability_pct:+.1%}"
        )

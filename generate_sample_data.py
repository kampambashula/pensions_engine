"""
generate_sample_data.py
-----------------------
Generates a realistic pension_analysis_ready.csv for testing the engine
when a real client CSV is not yet available.

Produces ~1,000 members with realistic actuarial distributions.
Scale up N_MEMBERS for load testing.

Usage:
    python generate_sample_data.py
    python generate_sample_data.py --members 5000
"""

from __future__ import annotations

import argparse
import random
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
rng  = np.random.default_rng(SEED)
random.seed(SEED)

OUT_PATH = Path(__file__).resolve().parent / "data" / "pension_analysis_ready.csv"

DEPARTMENTS = [
    "Finance", "Human Resources", "Operations", "Technology",
    "Legal & Compliance", "Risk Management", "Actuarial",
    "Client Services", "Investment", "Executive",
    "Research & Analytics", "Corporate Affairs",
]
SECTORS = ["Public", "Private", "Parastatal"]
PENSION_TYPES = ["Defined Benefit", "Defined Contribution", "Hybrid"]
FORMULA_TYPES = ["DB", "DC", "Hybrid"]
MORTALITY_TABLES = ["A1967-70", "PMA92", "PA(90)", "SAPS S2"]
STATUSES = ["Active", "Deferred", "Retired"]
EMPLOYMENT_STATUSES = ["Active", "Active", "Active", "Active", "Deferred", "Retired"]


def random_date(start: date, end: date) -> date:
    delta = (end - start).days
    return start + timedelta(days=int(rng.integers(0, delta)))


def generate_members(n: int) -> pd.DataFrame:
    today = date.today()
    records = []

    for i in range(1, n + 1):
        gender = rng.choice(["Male", "Female"], p=[0.55, 0.45])
        age = float(rng.integers(22, 65))
        dob = today - timedelta(days=int(age * 365.25) + int(rng.integers(-180, 180)))

        service = float(rng.integers(1, min(int(age - 20), 35) + 1))
        date_joined_employer = today - timedelta(days=int(service * 365.25))
        date_joined_scheme   = date_joined_employer + timedelta(days=int(rng.integers(0, 365)))

        retirement_age = float(rng.choice([55, 60, 65], p=[0.10, 0.30, 0.60]))
        years_to_ret   = max(0.0, retirement_age - age)
        nrd            = today + timedelta(days=int(years_to_ret * 365.25))

        dept   = rng.choice(DEPARTMENTS)
        sector = rng.choice(SECTORS)
        emp_status = rng.choice(EMPLOYMENT_STATUSES)

        # Salary bands
        salary_band = rng.choice(["junior", "mid", "senior", "exec"], p=[0.30, 0.40, 0.20, 0.10])
        if salary_band == "junior":
            basic = float(rng.integers(3_000, 8_000))
        elif salary_band == "mid":
            basic = float(rng.integers(8_000, 20_000))
        elif salary_band == "senior":
            basic = float(rng.integers(20_000, 50_000))
        else:
            basic = float(rng.integers(50_000, 150_000))

        housing    = round(basic * float(rng.uniform(0.10, 0.25)), 2)
        transport  = round(basic * float(rng.uniform(0.05, 0.12)), 2)
        total_sal  = round(basic + housing + transport, 2)
        pen_sal    = round(basic * float(rng.uniform(0.85, 1.00)), 2)
        annual_pen = round(pen_sal * 12, 2)

        ee_rate = float(rng.choice([0.05, 0.06, 0.07, 0.08, 0.10], p=[0.10, 0.20, 0.30, 0.25, 0.15]))
        er_rate = float(rng.choice([0.08, 0.10, 0.12, 0.15], p=[0.20, 0.35, 0.30, 0.15]))

        ee_monthly = round(pen_sal * ee_rate, 2)
        er_monthly = round(pen_sal * er_rate, 2)
        total_monthly = round(ee_monthly + er_monthly, 2)

        # Accumulated contributions — function of service
        ee_acc = round(ee_monthly * 12 * service * float(rng.uniform(0.8, 1.2)), 2)
        er_acc = round(er_monthly * 12 * service * float(rng.uniform(0.8, 1.2)), 2)
        total_acc = round(ee_acc + er_acc, 2)

        # Actuarial assumptions
        inv_ret   = round(float(rng.uniform(0.07, 0.10)), 4)
        sal_gr    = round(float(rng.uniform(0.04, 0.08)), 4)
        inflation = round(float(rng.uniform(0.05, 0.08)), 4)
        accrual   = float(rng.choice([0.0167, 0.02, 0.025], p=[0.20, 0.60, 0.20]))

        pen_type    = rng.choice(PENSION_TYPES)
        formula     = "DC" if "Contribution" in pen_type else ("Hybrid" if "Hybrid" in pen_type else "DB")
        mortality   = rng.choice(MORTALITY_TABLES)

        # Projected values
        proj_final = round(pen_sal * ((1 + sal_gr) ** years_to_ret), 2)
        proj_ben   = round(accrual * (service + years_to_ret) * proj_final, 2)

        records.append({
            "member_id":                       f"MBR{i:07d}",
            "employee_number":                 f"EMP{i:06d}",
            "nrc":                             f"{rng.integers(100000,999999)}/{rng.integers(10,99)}/{rng.integers(1,9)}",
            "first_name":                      f"Member{i}",
            "last_name":                       f"Surname{rng.integers(1,500)}",
            "gender":                          gender,
            "date_of_birth":                   dob.isoformat(),
            "date_joined_employer":            date_joined_employer.isoformat(),
            "date_joined_scheme":              date_joined_scheme.isoformat(),
            "employment_status":               emp_status,
            "department":                      dept,
            "sector":                          sector,
            "retirement_age":                  retirement_age,
            "normal_retirement_date":          nrd.isoformat(),
            "current_age":                     round(age, 2),
            "service_years":                   round(service, 2),
            "pensionable_service_years":       round(service * float(rng.uniform(0.90, 1.0)), 2),
            "years_to_retirement":             round(years_to_ret, 2),
            "basic_salary":                    basic,
            "housing_allowance":               housing,
            "transport_allowance":             transport,
            "total_monthly_salary":            total_sal,
            "pensionable_salary":              pen_sal,
            "annual_pensionable_salary":       annual_pen,
            "employee_contribution_rate":      ee_rate,
            "employer_contribution_rate":      er_rate,
            "employee_monthly_contribution":   ee_monthly,
            "employer_monthly_contribution":   er_monthly,
            "total_monthly_contribution":      total_monthly,
            "ee_accumulated_contributions":    ee_acc,
            "er_accumulated_contributions":    er_acc,
            "total_accumulated_contributions": total_acc,
            "investment_return_assumption":    inv_ret,
            "salary_growth_assumption":        sal_gr,
            "inflation_assumption":            inflation,
            "accrual_rate":                    accrual,
            "withdrawal_rate":                 round(float(rng.uniform(0.02, 0.05)), 4),
            "disability_rate":                 round(float(rng.uniform(0.002, 0.01)), 4),
            "spouse_percentage":               round(float(rng.uniform(0.50, 0.67)), 2),
            "mortality_table":                 mortality,
            "benefit_formula_type":            formula,
            "pension_type":                    pen_type,
            "projected_final_salary":          proj_final,
            "projected_pension_benefit":       proj_ben,
            "last_valuation_date":             date(today.year, 12, 31).isoformat(),
            "status":                          rng.choice(["Active", "Inactive"], p=[0.92, 0.08]),
        })

    return pd.DataFrame(records)


def main(n: int = 1_000) -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating {n:,} member records…")
    df = generate_members(n)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved → {OUT_PATH}  ({df.shape[0]:,} rows × {df.shape[1]} cols)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--members", type=int, default=1_000)
    args = parser.parse_args()
    main(args.members)

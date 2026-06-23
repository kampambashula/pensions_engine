# ◈ Pension Actuarial Intelligence Engine

**Production-grade pension valuation and actuarial analytics platform.**

Replaces slow Excel-based actuarial workflows with a fast, repeatable, database-backed engine and executive analytics dashboard.

---

## Architecture

```
pension_analysis_ready.csv
         ↓
 csv_to_sqlite.py          →  pension_engine.db (SQLite)
         ↓
 calculation_engine.py     →  PensionEngine API
         ↓
 app.py (Streamlit)        →  Executive Analytics Dashboard
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare your data

Place your cleaned CSV at:
```
data/pension_analysis_ready.csv
```

Or generate sample data for testing:
```bash
python generate_sample_data.py                  # 1,000 members (default)
python generate_sample_data.py --members 10000  # scale up
```

### 3. Import CSV → SQLite

```bash
python csv_to_sqlite.py
# or point to a specific CSV:
python csv_to_sqlite.py /path/to/your/pension_analysis_ready.csv
```

Expected output:
```
=== Import Summary ===
  db_path              pension_engine.db
  source_file          data/pension_analysis_ready.csv
  rows_total           105000
  rows_inserted        105000
  rows_skipped         0
  duration_secs        4.21
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

Open: http://localhost:8501

---

## File Structure

```
pension-calculation-engine/
│
├── app.py                      Streamlit dashboard (executive UI)
├── calculation_engine.py       Actuarial calculation core
├── csv_to_sqlite.py            CSV → SQLite importer
├── generate_sample_data.py     Sample data generator (testing)
├── requirements.txt
├── README.md
│
├── data/
│   └── pension_analysis_ready.csv    ← your clean input file
│
└── exports/                    ← reserved for report exports
```

---

## Dashboard Pages

| Page | Contents |
|---|---|
| **Executive Overview** | KPI cards, age distribution, retirement timeline, funding gauge |
| **Liability Analysis** | Department liability, funding waterfall, top-50 scatter |
| **Contribution Analysis** | Adequacy bands, funding ratio by department |
| **Scenario Testing** | Standard + custom scenario stress tests |
| **Risk Detection** | High-risk member flags, risk reason breakdown |
| **Member Explorer** | Filterable member liability table |

---

## Calculation Engine API

```python
from calculation_engine import PensionEngine

engine = PensionEngine()
engine.load()

# Portfolio summary
pm = engine.portfolio_metrics
print(f"Funding Ratio: {pm.funding_ratio:.1%}")
print(f"Total PV Liability: {pm.total_pv_liability:,.0f}")

# High-risk members
risk_df = engine.high_risk_members

# Standard scenario suite
scenarios = engine.run_scenarios()

# Custom scenario
result = engine.custom_scenario(
    "Stress Test",
    salary_growth=0.09,
    inflation=0.10,
    investment_return=0.05,
    retirement_age_delta=-2,
)
```

---

## Actuarial Methodology

### Benefit Projection

**Defined Benefit (DB):**
```
Projected Final Salary  = Pensionable Salary × (1 + salary_growth)^years_to_retirement
Projected Pension       = accrual_rate × total_service_at_retirement × projected_final_salary
```

**Defined Contribution (DC):**
```
FV = accumulated_fund × (1+r)^n  +  annual_contribution × [(1+r)^n - 1] / r
Annual Pension = FV / annuity_factor
```

### Present Value Liability

```
PV Liability = annual_pension × annuity_factor / (1 + investment_return)^years_to_retirement
```

### Funding Ratio

```
Funding Ratio = Total Accumulated Contributions / Total PV Liability
```

### Contribution Adequacy

```
Adequacy Ratio = Member Accumulated Fund / PV Liability
  ≥ 1.00  →  Fully funded
  0.80–1.00  →  Near-funded
  < 0.50  →  High risk
```

### High-Risk Detection Criteria

- Contribution adequacy ratio < 20%
- Near retirement (≤5 years) with adequacy < 50%
- Zero accumulated contributions
- Single member liability > 2% of total scheme liability

---

## Database Schema

Main table: `members`

Key indexed fields:
- `member_id` (UNIQUE)
- `current_age`
- `years_to_retirement`
- `department`
- `employment_status`
- `gender`

Supporting tables:
- `scenario_runs` — persisted scenario history
- `import_log` — data lineage and import audit trail

---

## Standard Scenario Suite

| Scenario | Description |
|---|---|
| Base Case | Default assumptions |
| High Inflation +2% | Inflation sensitivity |
| Low Returns -2% | Investment return stress |
| High Salary Growth +3% | Salary escalation sensitivity |
| Early Retirement -2yr | Early retirement concentration |
| Late Retirement +2yr | Deferred retirement benefit |
| Stress Test | Combined adverse scenario |

---

## Migration Path

### → PostgreSQL

Replace SQLite connection string in `csv_to_sqlite.py` and `calculation_engine.py`:

```python
# SQLite (current)
conn = sqlite3.connect("pension_engine.db")

# PostgreSQL (future)
import psycopg2
conn = psycopg2.connect("host=... dbname=pension_engine user=... password=...")
```

Schema DDL is ANSI-compatible. No SQLite-specific syntax used.

### → Django Integration

`PensionEngine` is designed as a stateless service class — drop it into a Django view or management command with no changes:

```python
# views.py
from calculation_engine import PensionEngine

def dashboard_api(request):
    engine = PensionEngine(db_path=settings.PENSION_DB_PATH)
    engine.load()
    return JsonResponse({"funding_ratio": engine.portfolio_metrics.funding_ratio})
```

---

## Performance Notes

- SQLite with WAL journal mode handles 100,000+ row imports in < 10 seconds
- WAL + memory cache configured for fast analytical reads
- `@st.cache_resource` ensures the engine loads once per Streamlit session
- All scenario runs operate on in-memory DataFrames — no repeated DB hits

---

*Pension Actuarial Intelligence Engine — Consulting Grade v1.0*

"""
app.py
------
Pension Actuarial Intelligence Dashboard
Executive-grade Streamlit analytics interface.

Designed for: pension managers, trustees, actuaries, consulting partners.
Aesthetic: Bloomberg Terminal × Executive Consulting — dark, precise, data-dense.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from calculation_engine import PensionEngine, DEFAULT_ASSUMPTIONS, PortfolioMetrics
from actuarial_tables import (
    mortality_table_summary, ill_health_table_summary,
    annuity_factor_grid, get_mortality_table, get_ill_health_table,
)
from assets_loader import load_fund_financials, compute_solvency_position, FundFinancials

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Pension Actuarial Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design system — Deloitte · Palantir · McKinsey consulting aesthetic
# Light, precise, institutional. Clean white surfaces, navy authority,
# Deloitte green accents, surgical typography.
# ---------------------------------------------------------------------------

DARK_BG        = "#FFFFFF"          # pure white canvas
PANEL_BG       = "#FAFBFC"          # off-white card surface
PANEL_BORDER   = "#E2E6EA"          # cool light grey border
ACCENT_AMBER   = "#86BC25"          # Deloitte signature green
ACCENT_BLUE    = "#002776"          # McKinsey/consulting deep navy
ACCENT_GREEN   = "#00A551"          # mid-green positive
ACCENT_RED     = "#C0392B"          # risk / alert red
ACCENT_SLATE   = "#6B7684"          # body text secondary
TEXT_PRIMARY   = "#1A2332"          # near-black navy text
TEXT_SECONDARY = "#6B7684"          # muted slate secondary
GRID_COLOR     = "#EDF0F3"          # very light grid
CHART_BG       = "#FFFFFF"          # white chart background
NAVY_SIDEBAR   = "#001F5B"          # deep Deloitte/McKinsey nav navy
SIDEBAR_TEXT   = "#FFFFFF"
SIDEBAR_ACCENT = "#86BC25"

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(color=TEXT_PRIMARY, family="'DM Sans', sans-serif", size=11),
        colorway=[ACCENT_BLUE, ACCENT_AMBER, ACCENT_GREEN, ACCENT_RED, "#0369A1", "#7C3AED"],
        xaxis=dict(gridcolor=GRID_COLOR, linecolor=PANEL_BORDER, zerolinecolor=PANEL_BORDER),
        yaxis=dict(gridcolor=GRID_COLOR, linecolor=PANEL_BORDER, zerolinecolor=PANEL_BORDER),
        margin=dict(l=40, r=20, t=44, b=40),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor=PANEL_BORDER,
                    borderwidth=1),
    )
)

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Serif+Display&display=swap');

/* ── Reset & base ─────────────────────────────────────────────────── */
html, body, [class*="css"] {{
    background-color: #FFFFFF !important;
    color: {TEXT_PRIMARY};
    font-family: 'DM Sans', sans-serif;
    -webkit-font-smoothing: antialiased;
}}

/* ── Sidebar — deep navy consulting nav ───────────────────────────── */
section[data-testid="stSidebar"] {{
    background: linear-gradient(180deg, {NAVY_SIDEBAR} 0%, #001444 100%) !important;
    border-right: none !important;
    box-shadow: 4px 0 20px rgba(0,31,91,0.12);
}}
section[data-testid="stSidebar"] * {{
    color: rgba(255,255,255,0.88) !important;
}}
section[data-testid="stSidebar"] .stRadio > div {{
    gap: 0.1rem;
}}
section[data-testid="stSidebar"] .stRadio label {{
    padding: 0.55rem 0.75rem !important;
    border-radius: 6px !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    letter-spacing: 0.01em !important;
    transition: background 0.15s ease !important;
    cursor: pointer !important;
}}
section[data-testid="stSidebar"] .stRadio label:hover {{
    background: rgba(134,188,37,0.15) !important;
}}
section[data-testid="stSidebar"] .stRadio [data-checked="true"] label,
section[data-testid="stSidebar"] .stRadio input:checked + div {{
    background: rgba(134,188,37,0.2) !important;
    border-left: 3px solid {SIDEBAR_ACCENT} !important;
    color: #FFFFFF !important;
}}

/* ── Main container ───────────────────────────────────────────────── */
.main .block-container {{
    padding: 2rem 2.5rem 3rem 2.5rem;
    max-width: 100%;
    background: #FFFFFF;
}}

/* ── Typography ───────────────────────────────────────────────────── */
h1, h2, h3 {{
    font-family: 'DM Serif Display', serif !important;
    font-weight: 400 !important;
    color: {TEXT_PRIMARY} !important;
    letter-spacing: -0.02em;
}}

/* ── KPI Cards — clean white elevated ────────────────────────────── */
.kpi-card {{
    background: #FFFFFF;
    border: 1px solid {PANEL_BORDER};
    border-top: 4px solid {ACCENT_AMBER};
    border-radius: 8px;
    padding: 1.4rem 1.6rem 1.2rem 1.6rem;
    margin-bottom: 0;
    box-shadow: 0 1px 4px rgba(26,35,50,0.06), 0 4px 12px rgba(26,35,50,0.04);
    transition: box-shadow 0.2s ease;
}}
.kpi-card:hover {{
    box-shadow: 0 2px 8px rgba(26,35,50,0.10), 0 8px 24px rgba(26,35,50,0.06);
}}
.kpi-label {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
    color: {TEXT_SECONDARY};
    margin-bottom: 0.5rem;
}}
.kpi-value {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem;
    font-weight: 400;
    color: {TEXT_PRIMARY};
    line-height: 1;
    letter-spacing: -0.02em;
}}
.kpi-sub {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.72rem;
    color: {TEXT_SECONDARY};
    margin-top: 0.4rem;
    font-weight: 400;
}}
.kpi-card.alert {{
    border-top-color: {ACCENT_RED};
}}
.kpi-card.alert .kpi-value {{
    color: {ACCENT_RED};
}}
.kpi-card.success {{
    border-top-color: {ACCENT_GREEN};
}}
.kpi-card.success .kpi-value {{
    color: {ACCENT_GREEN};
}}
.kpi-card.info {{
    border-top-color: {ACCENT_BLUE};
}}

/* ── Section headers ──────────────────────────────────────────────── */
.section-header {{
    font-family: 'DM Sans', sans-serif;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {ACCENT_AMBER};
    border-bottom: 2px solid {PANEL_BORDER};
    padding-bottom: 0.6rem;
    margin: 2rem 0 1.25rem 0;
}}

/* ── Data tables ──────────────────────────────────────────────────── */
.dataframe {{
    background: #FFFFFF !important;
    color: {TEXT_PRIMARY} !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    border: 1px solid {PANEL_BORDER} !important;
}}
.dataframe thead th {{
    background: #F4F6F9 !important;
    color: {TEXT_SECONDARY} !important;
    font-weight: 600 !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}

/* ── Streamlit native metric ──────────────────────────────────────── */
div[data-testid="metric-container"] {{
    background: #FFFFFF;
    border: 1px solid {PANEL_BORDER};
    padding: 1rem 1.25rem;
    border-radius: 8px;
    box-shadow: 0 1px 4px rgba(26,35,50,0.06);
}}

/* ── Sliders ──────────────────────────────────────────────────────── */
.stSlider > div > div > div > div {{
    background: {ACCENT_AMBER};
}}

/* ── Alerts / banners ─────────────────────────────────────────────── */
.stAlert {{
    background: #FFF8F8;
    border: 1px solid {ACCENT_RED};
    color: {TEXT_PRIMARY};
    border-radius: 6px;
}}
.risk-banner {{
    background: #FEF2F2;
    border: 1px solid #FECACA;
    border-left: 4px solid {ACCENT_RED};
    padding: 0.8rem 1.1rem;
    border-radius: 0 6px 6px 0;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    color: {TEXT_PRIMARY};
    margin: 0.5rem 0;
}}
.info-banner {{
    background: #F0F7FF;
    border: 1px solid #BFDBFE;
    border-left: 4px solid {ACCENT_BLUE};
    padding: 0.8rem 1.1rem;
    border-radius: 0 6px 6px 0;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    color: {TEXT_PRIMARY};
    margin: 0.5rem 0;
}}

/* ── Selectbox ────────────────────────────────────────────────────── */
.stSelectbox > div > div {{
    background: #FFFFFF;
    border-color: {PANEL_BORDER};
    border-radius: 6px;
    color: {TEXT_PRIMARY};
}}

/* ── Buttons ──────────────────────────────────────────────────────── */
.stButton > button {{
    background: {ACCENT_BLUE} !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.6rem 1.5rem !important;
    transition: background 0.15s ease !important;
}}
.stButton > button:hover {{
    background: #001F5B !important;
}}

/* ── Divider ──────────────────────────────────────────────────────── */
hr {{
    border: none;
    border-top: 1px solid {PANEL_BORDER};
    margin: 1rem 0;
}}

/* ── Scrollbar ────────────────────────────────────────────────────── */
::-webkit-scrollbar {{ width: 5px; height: 5px; }}
::-webkit-scrollbar-track {{ background: #F4F6F9; }}
::-webkit-scrollbar-thumb {{ background: #C8D0DB; border-radius: 3px; }}
</style>
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_currency(v: float, abbrev: bool = True) -> str:
    if abbrev:
        if abs(v) >= 1e9:
            return f"{v/1e9:.2f}B"
        if abs(v) >= 1e6:
            return f"{v/1e6:.1f}M"
        if abs(v) >= 1e3:
            return f"{v/1e3:.0f}K"
    return f"{v:,.0f}"


def fmt_pct(v: float) -> str:
    return f"{v:.1%}" if pd.notna(v) else "N/A"


def kpi_html(label: str, value: str, sub: str = "", variant: str = "") -> str:
    cls = f"kpi-card {variant}".strip()
    return f"""
    <div class="{cls}">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {'<div class="kpi-sub">' + sub + '</div>' if sub else ''}
    </div>
    """


def section(title: str) -> None:
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


def apply_template(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_TEMPLATE["layout"].to_plotly_json())
    return fig


# ---------------------------------------------------------------------------
# Data loading — cached
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Initialising pension engine…")
def get_engine() -> PensionEngine:
    engine = PensionEngine()
    engine.load()
    return engine


@st.cache_data(ttl=300, show_spinner=False)
def load_scenario_results(_engine: PensionEngine) -> list:
    return _engine.run_scenarios()


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def chart_age_distribution(df_age: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_age["age_band"].astype(str),
        y=df_age["member_count"],
        name="Members",
        marker_color=ACCENT_AMBER,
        opacity=0.9,
    ))
    fig.add_trace(go.Scatter(
        x=df_age["age_band"].astype(str),
        y=df_age["total_liability"],
        name="PV Liability",
        yaxis="y2",
        line=dict(color=ACCENT_BLUE, width=2),
        mode="lines+markers",
        marker=dict(size=6),
    ))
    fig.update_layout(
        title="Age Distribution & Liability Profile",
        yaxis=dict(title="Members", gridcolor=GRID_COLOR),
        yaxis2=dict(title="PV Liability", overlaying="y", side="right", gridcolor=GRID_COLOR),
        barmode="group",
        legend=dict(orientation="h", y=1.1),
        height=350,
    )
    return apply_template(fig)


def chart_contribution_adequacy(df_bands: pd.DataFrame) -> go.Figure:
    colors = [ACCENT_RED, ACCENT_RED, ACCENT_AMBER, ACCENT_AMBER, ACCENT_GREEN, ACCENT_GREEN]
    fig = go.Figure(go.Bar(
        x=df_bands["adequacy_band"],
        y=df_bands["member_count"],
        marker_color=colors[:len(df_bands)],
        opacity=0.85,
    ))
    fig.update_layout(
        title="Contribution Adequacy Distribution",
        xaxis_title="Adequacy Band (Accumulated / PV Liability)",
        yaxis_title="Members",
        height=320,
    )
    return apply_template(fig)


def chart_retirement_timeline(df_rt: pd.DataFrame) -> go.Figure:
    df_rt = df_rt[df_rt["retirement_year"].between(
        pd.Timestamp.now().year,
        pd.Timestamp.now().year + 20
    )]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_rt["retirement_year"],
        y=df_rt["member_count"],
        name="Members Retiring",
        marker_color=ACCENT_AMBER,
        opacity=0.85,
    ))
    fig.add_trace(go.Scatter(
        x=df_rt["retirement_year"],
        y=df_rt["total_liability"].cumsum(),
        name="Cumulative Liability",
        yaxis="y2",
        line=dict(color=ACCENT_BLUE, width=2, dash="dot"),
        mode="lines",
    ))
    fig.update_layout(
        title="Retirement Timeline (20-Year Projection)",
        yaxis=dict(title="Members"),
        yaxis2=dict(title="Cumulative Liability", overlaying="y", side="right"),
        height=350,
    )
    return apply_template(fig)


def chart_department_liability(df_dept: pd.DataFrame) -> go.Figure:
    top = df_dept.nlargest(15, "total_liability")
    fig = go.Figure(go.Bar(
        x=top["total_liability"],
        y=top["department"],
        orientation="h",
        marker=dict(
            color=top["total_liability"],
            colorscale=[[0, "#E8F4FD"], [0.5, ACCENT_AMBER], [1, ACCENT_BLUE]],
            showscale=False,
        ),
        text=top["total_liability"].apply(lambda v: fmt_currency(v)),
        textposition="outside",
        textfont=dict(size=10),
    ))
    fig.update_layout(
        title="Liability Concentration by Department",
        xaxis_title="Present Value Liability",
        height=max(280, len(top) * 28),
        margin=dict(l=120),
    )
    return apply_template(fig)


def chart_scenario_comparison(scenarios: list) -> go.Figure:
    names = [s.scenario_name for s in scenarios]
    liabs = [s.total_pv_liability for s in scenarios]
    ratios = [s.funding_ratio for s in scenarios]

    base_liab = liabs[0] if liabs else 1

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="PV Liability",
        x=names,
        y=liabs,
        marker_color=[ACCENT_RED if l > base_liab else ACCENT_GREEN for l in liabs],
        opacity=0.8,
    ))
    fig.add_trace(go.Scatter(
        name="Funding Ratio",
        x=names,
        y=[r * 100 for r in ratios],
        yaxis="y2",
        mode="lines+markers",
        line=dict(color=ACCENT_AMBER, width=2),
        marker=dict(size=8),
    ))
    fig.update_layout(
        title="Scenario Analysis — Liability & Funding Impact",
        yaxis=dict(title="PV Liability"),
        yaxis2=dict(title="Funding Ratio (%)", overlaying="y", side="right"),
        xaxis_tickangle=-25,
        height=380,
        legend=dict(orientation="h", y=1.08),
    )
    return apply_template(fig)


def chart_funding_gauge(funding_ratio: float) -> go.Figure:
    pct = min(funding_ratio * 100, 150) if pd.notna(funding_ratio) else 0
    color = (
        ACCENT_GREEN if pct >= 100
        else ACCENT_AMBER if pct >= 80
        else ACCENT_RED
    )
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        number=dict(suffix="%", font=dict(size=28, color=TEXT_PRIMARY, family="DM Sans")),
        delta=dict(reference=100, valueformat=".1f", suffix="%"),
        gauge=dict(
            axis=dict(
                range=[0, 150],
                tickwidth=1,
                tickcolor=TEXT_SECONDARY,
                tickfont=dict(size=10, family="DM Sans"),
            ),
            bar=dict(color=color, thickness=0.25),
            bgcolor='#FAFBFC',
            borderwidth=1,
            bordercolor=PANEL_BORDER,
            steps=[
                dict(range=[0, 80],   color="#FEF2F2"),
                dict(range=[80, 100], color="#FFFBEB"),
                dict(range=[100, 150],color="#F0FDF4"),
            ],
            threshold=dict(
                line=dict(color=ACCENT_BLUE, width=2),
                thickness=0.75,
                value=100,
            ),
        ),
        title=dict(text="Funding Ratio", font=dict(color=TEXT_SECONDARY, size=12, family="DM Sans")),
        domain=dict(x=[0, 1], y=[0, 1]),
    ))
    fig.update_layout(paper_bgcolor='#FFFFFF', height=260, margin=dict(l=20, r=20, t=40, b=0))
    return fig


def chart_gender_split(df_gender: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Pie(
        labels=df_gender["gender"],
        values=df_gender["count"],
        hole=0.65,
        marker=dict(colors=[ACCENT_BLUE, ACCENT_AMBER, ACCENT_SLATE]),
        textinfo="label+percent",
        textfont=dict(size=11, family="DM Sans"),
    ))
    fig.update_layout(
        title="Gender Distribution",
        height=280,
        showlegend=False,
        annotations=[dict(text="MEMBERS", x=0.5, y=0.5, showarrow=False,
                         font=dict(size=10, color=TEXT_SECONDARY, family="DM Sans"))],
    )
    return apply_template(fig)


def chart_contribution_waterfall(pm: PortfolioMetrics) -> go.Figure:
    fig = go.Figure(go.Waterfall(
        name="Funding Position",
        orientation="v",
        measure=["relative", "relative", "total", "relative", "total"],
        x=["EE Accumulated", "ER Accumulated", "Total Fund", "PV Liability", "Surplus / Deficit"],
        y=[
            pm.total_ee_accumulated,
            pm.total_er_accumulated,
            0,
            -pm.total_pv_liability,
            0,
        ],
        connector=dict(line=dict(color=PANEL_BORDER, width=1)),
        increasing=dict(marker=dict(color=ACCENT_GREEN)),
        decreasing=dict(marker=dict(color=ACCENT_RED)),
        totals=dict(marker=dict(color=ACCENT_AMBER)),
        text=[
            fmt_currency(pm.total_ee_accumulated),
            fmt_currency(pm.total_er_accumulated),
            fmt_currency(pm.total_accumulated),
            fmt_currency(pm.total_pv_liability),
            fmt_currency(pm.total_accumulated - pm.total_pv_liability),
        ],
        textposition="outside",
        textfont=dict(size=9, family="DM Sans"),
    ))
    fig.update_layout(
        title="Funding Waterfall",
        yaxis_title="Amount",
        height=340,
        showlegend=False,
    )
    return apply_template(fig)


def chart_top_liability_scatter(df: pd.DataFrame) -> go.Figure:
    top = df.nlargest(50, "pv_liability").copy()
    top["name_label"] = top["first_name"].fillna("") + " " + top["last_name"].fillna("")
    top["name_label"] = top["name_label"].str.strip().replace("", "Unknown")

    fig = go.Figure(go.Scatter(
        x=top["current_age"],
        y=top["pv_liability"],
        mode="markers",
        marker=dict(
            size=np.sqrt(top["annual_pensionable_salary"].clip(lower=0) / 1000).clip(4, 20),
            color=top["years_to_retirement"],
            colorscale=[[0, ACCENT_RED], [0.5, ACCENT_AMBER], [1, ACCENT_GREEN]],
            showscale=True,
            colorbar=dict(title="Yrs to Ret.", thickness=12, len=0.7,
                          tickfont=dict(size=9), title_font=dict(size=9)),
            opacity=0.8,
            line=dict(width=0.5, color=PANEL_BORDER),
        ),
        text=top["name_label"],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Age: %{x:.1f}<br>"
            "PV Liability: %{y:,.0f}<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        title="Top-50 Liability Members (bubble = salary)",
        xaxis_title="Current Age",
        yaxis_title="PV Liability",
        height=380,
    )
    return apply_template(fig)


# ---------------------------------------------------------------------------
# Page sections
# ---------------------------------------------------------------------------

def render_header() -> None:
    from datetime import date
    col_logo, col_title, col_meta = st.columns([1, 5, 2])
    with col_title:
        st.markdown(
            f'<div style="display:flex; align-items:baseline; gap:0.75rem;">'
            f'<h1 style="font-family:\'DM Serif Display\',serif; font-size:1.7rem; '
            f'font-weight:400; margin:0; color:{TEXT_PRIMARY}; letter-spacing:-0.02em;">'
            f'Pension Actuarial Intelligence</h1>'
            f'<span style="font-family:\'DM Sans\',sans-serif; font-size:0.65rem; '
            f'font-weight:700; letter-spacing:0.12em; text-transform:uppercase; '
            f'color:{ACCENT_AMBER}; background:{ACCENT_AMBER}18; padding:0.2rem 0.5rem; '
            f'border-radius:3px;">PLATFORM</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<p style="font-family:\'DM Sans\',sans-serif; font-size:0.78rem; '
            f'color:{TEXT_SECONDARY}; margin:0.25rem 0 0 0; font-weight:400;">'
            f'Actuarial Valuation · Liability Analytics · Decision Support</p>',
            unsafe_allow_html=True,
        )
    with col_meta:
        st.markdown(
            f'<div style="text-align:right; padding-top:0.25rem;">'
            f'<div style="font-family:\'DM Sans\',sans-serif; font-size:0.6rem; '
            f'font-weight:600; letter-spacing:0.1em; text-transform:uppercase; '
            f'color:{TEXT_SECONDARY};">Valuation Date</div>'
            f'<div style="font-family:\'DM Serif Display\',serif; font-size:1.1rem; '
            f'color:{ACCENT_BLUE}; font-weight:400;">'
            f'{date.today().strftime("%d %B %Y")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div style="height:2px; background:linear-gradient(90deg, {ACCENT_BLUE} 0%, '
        f'{ACCENT_AMBER} 60%, transparent 100%); margin:1rem 0 0.5rem 0;"></div>',
        unsafe_allow_html=True,
    )


def render_kpi_row(pm: PortfolioMetrics) -> None:
    section("EXECUTIVE SUMMARY — KEY METRICS")
    cols = st.columns(6)

    funding_variant = (
        "success" if pm.funding_ratio >= 1.0
        else "alert" if pm.funding_ratio < 0.80
        else ""
    )

    kpis = [
        ("TOTAL MEMBERS",         f"{pm.total_members:,}",                       f"Active: {pm.active_members:,}",               ""),
        ("TOTAL FUND (ACCUM.)",    fmt_currency(pm.total_accumulated),             f"EE: {fmt_currency(pm.total_ee_accumulated)}",  "info"),
        ("PV TOTAL LIABILITY",     fmt_currency(pm.total_pv_liability),            "Present value basis",                          "alert"),
        ("FUNDING RATIO",          fmt_pct(pm.funding_ratio),                      "Assets / Liabilities",                         funding_variant),
        ("NEAR RETIREMENT",        f"{pm.near_retirement_count:,}",                "Within 5 years of NRD",                        "alert" if pm.near_retirement_count > pm.total_members * 0.10 else ""),
        ("HIGH-RISK MEMBERS",      f"{pm.high_risk_count:,}",                      "Requires attention",                           "alert" if pm.high_risk_count > 0 else "success"),
    ]

    for col, (label, value, sub, variant) in zip(cols, kpis):
        with col:
            st.markdown(kpi_html(label, value, sub, variant), unsafe_allow_html=True)


def render_secondary_kpis(pm: PortfolioMetrics) -> None:
    cols = st.columns(4)
    secondary = [
        ("AVG MEMBER AGE",         f"{pm.avg_age:.1f} yrs",      ""),
        ("AVG SERVICE",            f"{pm.avg_service:.1f} yrs",   ""),
        ("ANNUAL PAYROLL",         fmt_currency(pm.total_annual_payroll), ""),
        ("ANNUAL CONTRIBUTIONS",   fmt_currency(pm.total_annual_contributions), ""),
    ]
    for col, (label, value, sub) in zip(cols, secondary):
        with col:
            st.markdown(kpi_html(label, value, sub, "info"), unsafe_allow_html=True)


def render_overview_charts(engine: PensionEngine, pm: PortfolioMetrics) -> None:
    section("PORTFOLIO OVERVIEW")
    c1, c2, c3 = st.columns([2, 2, 1.2])

    with c1:
        df_age = engine.get_age_distribution()
        st.plotly_chart(chart_age_distribution(df_age), use_container_width=True, config={"displayModeBar": False})

    with c2:
        df_rt = engine.get_retirement_timeline()
        st.plotly_chart(chart_retirement_timeline(df_rt), use_container_width=True, config={"displayModeBar": False})

    with c3:
        st.plotly_chart(chart_funding_gauge(pm.funding_ratio), use_container_width=True, config={"displayModeBar": False})
        df_gender = engine.get_gender_distribution()
        st.plotly_chart(chart_gender_split(df_gender), use_container_width=True, config={"displayModeBar": False})


def render_liability_analysis(engine: PensionEngine) -> None:
    section("LIABILITY & FUNDING ANALYSIS")
    c1, c2 = st.columns([3, 2])

    with c1:
        df_dept = engine.get_department_liability()
        st.plotly_chart(chart_department_liability(df_dept), use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.plotly_chart(chart_contribution_waterfall(engine.portfolio_metrics), use_container_width=True, config={"displayModeBar": False})

    st.plotly_chart(chart_top_liability_scatter(engine.members), use_container_width=True, config={"displayModeBar": False})


def render_contribution_analysis(engine: PensionEngine) -> None:
    section("CONTRIBUTION ADEQUACY")
    c1, c2 = st.columns(2)

    with c1:
        df_bands = engine.get_contribution_adequacy_bands()
        st.plotly_chart(chart_contribution_adequacy(df_bands), use_container_width=True, config={"displayModeBar": False})

    with c2:
        df_fund_dept = engine.get_funding_by_department()
        top_dept = df_fund_dept.nlargest(12, "total_liability")

        fig = px.bar(
            top_dept,
            x="department",
            y="funding_ratio",
            color="funding_ratio",
            color_continuous_scale=[[0, ACCENT_RED], [0.8, ACCENT_AMBER], [1, ACCENT_GREEN]],
            title="Funding Ratio by Department",
            labels={"funding_ratio": "Funding Ratio", "department": ""},
        )
        fig.add_hline(y=1.0, line_dash="dash", line_color="white", opacity=0.5,
                     annotation_text="100%", annotation_font_size=9)
        fig.update_layout(height=320, coloraxis_showscale=False, xaxis_tickangle=-30)
        apply_template(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_scenario_analysis(engine: PensionEngine) -> None:
    section("SCENARIO ANALYSIS — ACTUARIAL STRESS TESTING")

    col_presets, col_custom = st.columns([3, 2])

    with col_presets:
        with st.spinner("Running actuarial scenarios…"):
            scenarios = load_scenario_results(engine)
        st.plotly_chart(chart_scenario_comparison(scenarios), use_container_width=True, config={"displayModeBar": False})

    with col_custom:
        st.markdown(
            f'<p style="font-family:\'DM Sans\',monospace; font-size:0.7rem; '
            f'color:{ACCENT_AMBER}; letter-spacing:0.1em; margin-bottom:0.75rem;">CUSTOM SCENARIO</p>',
            unsafe_allow_html=True,
        )
        sg   = st.slider("Salary Growth (%)",       1.0, 15.0, float(DEFAULT_ASSUMPTIONS["salary_growth"]   * 100), 0.5) / 100
        ir   = st.slider("Investment Return (%)",   1.0, 15.0, float(DEFAULT_ASSUMPTIONS["investment_return"] * 100), 0.5) / 100
        inf  = st.slider("Inflation (%)",           1.0, 15.0, float(DEFAULT_ASSUMPTIONS["inflation"]       * 100), 0.5) / 100
        rad  = st.slider("Retirement Age Shift (yr)", -5, 5, 0, 1)

        if st.button("▶ RUN SCENARIO", use_container_width=True):
            result = engine.custom_scenario(
                "Custom",
                salary_growth=sg, inflation=inf,
                investment_return=ir, retirement_age_delta=float(rad),
            )
            base = scenarios[0]
            delta_l = (result.total_pv_liability - base.total_pv_liability) / base.total_pv_liability
            delta_f = result.funding_ratio - base.funding_ratio

            st.markdown(
                f"""
                <div style="background:#F0F7FF; border:1px solid #BFDBFE;
                            border-left:4px solid {ACCENT_BLUE}; padding:1.1rem 1.25rem;
                            border-radius:0 8px 8px 0; margin-top:0.75rem;
                            font-family:'DM Sans',sans-serif; font-size:0.8rem;">
                  <div style="font-size:0.6rem; font-weight:700; letter-spacing:0.12em;
                              text-transform:uppercase; color:{ACCENT_BLUE}; margin-bottom:0.75rem;">
                    Custom Scenario Result</div>
                  <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.4rem 0.2rem;">
                    <div style="color:{TEXT_SECONDARY};">PV Liability</div>
                    <div style="color:{TEXT_PRIMARY}; font-weight:600; text-align:right;">{fmt_currency(result.total_pv_liability)}</div>
                    <div style="color:{TEXT_SECONDARY};">Change vs Base</div>
                    <div style="color:{'#C0392B' if delta_l > 0 else '#00A551'}; font-weight:600; text-align:right;">{delta_l:+.1%}</div>
                    <div style="color:{TEXT_SECONDARY};">Funding Ratio</div>
                    <div style="color:{ACCENT_BLUE}; font-weight:600; text-align:right;">{result.funding_ratio:.1%}</div>
                    <div style="color:{TEXT_SECONDARY};">Change vs Base</div>
                    <div style="color:{'#00A551' if delta_f > 0 else '#C0392B'}; font-weight:600; text-align:right;">{delta_f:+.2%}</div>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # Scenario table
    df_sc = pd.DataFrame([
        {
            "Scenario":         s.scenario_name,
            "Salary Growth":    f"{s.salary_growth:.1%}",
            "Investment Ret.":  f"{s.investment_return:.1%}",
            "Inflation":        f"{s.inflation:.1%}",
            "PV Liability":     fmt_currency(s.total_pv_liability),
            "Funding Ratio":    f"{s.funding_ratio:.1%}",
            "Δ Liability":      f"{s.delta_liability_pct:+.1%}",
        }
        for s in scenarios
    ])
    st.dataframe(
        df_sc,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Δ Liability": st.column_config.TextColumn("Δ vs Base"),
        },
    )


def render_risk_analysis(engine: PensionEngine) -> None:
    section("HIGH-RISK MEMBER DETECTION")

    risk_df = engine.high_risk_members
    pm = engine.portfolio_metrics

    if len(risk_df) == 0:
        st.markdown(
            '<div class="info-banner">✓ No high-risk members identified under current thresholds.</div>',
            unsafe_allow_html=True,
        )
        return

    pct_risk = len(risk_df) / pm.total_members
    risk_liability_pct = risk_df["pv_liability"].sum() / pm.total_pv_liability if pm.total_pv_liability > 0 else 0

    col_a, col_b = st.columns([1, 3])
    with col_a:
        st.markdown(kpi_html("HIGH-RISK MEMBERS", f"{len(risk_df):,}", f"{pct_risk:.1%} of scheme", "alert"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(kpi_html("RISK LIABILITY", fmt_currency(risk_df['pv_liability'].sum()), f"{risk_liability_pct:.1%} of total PVL", "alert"), unsafe_allow_html=True)

    with col_b:
        risk_counts = risk_df["risk_reason"].str.split("; ").explode().value_counts().reset_index()
        risk_counts.columns = ["Reason", "Count"]
        fig = go.Figure(go.Bar(
            x=risk_counts["Count"],
            y=risk_counts["Reason"],
            orientation="h",
            marker_color=ACCENT_RED,
            opacity=0.85,
        ))
        fig.update_layout(title="Risk Flags Distribution", height=260, margin=dict(l=180))
        apply_template(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    display_cols = [
        "member_id", "first_name", "last_name", "department",
        "current_age", "years_to_retirement",
        "total_accumulated_contributions", "pv_liability",
        "contribution_adequacy_ratio", "risk_reason",
    ]
    available = [c for c in display_cols if c in risk_df.columns]
    show = risk_df[available].head(50).copy()

    # Format for display
    for col in ["total_accumulated_contributions", "pv_liability"]:
        if col in show.columns:
            show[col] = show[col].apply(fmt_currency)
    if "contribution_adequacy_ratio" in show.columns:
        show["contribution_adequacy_ratio"] = show["contribution_adequacy_ratio"].apply(
            lambda v: f"{v:.1%}" if pd.notna(v) else "N/A"
        )

    st.dataframe(show, use_container_width=True, hide_index=True)


def render_member_explorer(engine: PensionEngine) -> None:
    section("MEMBER LIABILITY EXPLORER")
    top_n = engine.get_top_liability_members(50)

    c1, c2 = st.columns([1, 3])
    with c1:
        dept_options = ["All Departments"] + sorted(
            engine.members["department"].dropna().unique().tolist()
        )
        selected_dept = st.selectbox("Filter by Department", dept_options)
        age_min, age_max = st.slider(
            "Age Range",
            min_value=int(engine.members["current_age"].min() or 18),
            max_value=int(engine.members["current_age"].max() or 65),
            value=(int(engine.members["current_age"].min() or 18),
                   int(engine.members["current_age"].max() or 65)),
        )

    with c2:
        df_view = engine.members.copy()
        if selected_dept != "All Departments":
            df_view = df_view[df_view["department"] == selected_dept]
        df_view = df_view[
            df_view["current_age"].between(age_min, age_max)
        ]

        display = df_view[[
            "member_id", "first_name", "last_name", "gender", "department",
            "current_age", "service_years", "years_to_retirement",
            "annual_pensionable_salary", "total_accumulated_contributions",
            "pv_liability", "contribution_adequacy_ratio",
        ]].nlargest(100, "pv_liability").copy()

        for col in ["annual_pensionable_salary", "total_accumulated_contributions", "pv_liability"]:
            if col in display.columns:
                display[col] = display[col].apply(fmt_currency)
        if "contribution_adequacy_ratio" in display.columns:
            display["contribution_adequacy_ratio"] = display["contribution_adequacy_ratio"].apply(
                lambda v: f"{v:.1%}" if pd.notna(v) else "N/A"
            )
        if "current_age" in display.columns:
            display["current_age"] = display["current_age"].round(1)

        st.caption(f"Showing top 100 by liability — {len(df_view):,} members match filter")
        st.dataframe(display, use_container_width=True, hide_index=True)



# ---------------------------------------------------------------------------
# Actuarial Output Tables
# ---------------------------------------------------------------------------

def _compute_output_tables(engine: "PensionEngine") -> dict:
    """
    Derive Future Service Liability and Past Service Liability tables
    from enriched member data, split by sub-fund.

    Sub-fund assignment:
      If the CSV contains a "sub_fund" column with values A/B that is used directly.
      Otherwise members are split deterministically by member_id last digit parity
      (even = Sub Fund A, odd = Sub Fund B).  Replace with real fund codes in production.
    """
    df = engine.members.copy()
    pm = engine.portfolio_metrics

    # Sub-fund assignment
    if "sub_fund" in df.columns:
        df["_sf"] = df["sub_fund"].str.strip().str.upper().map(
            lambda v: "A" if v == "A" else "B"
        )
    else:
        df["_sf"] = df["member_id"].apply(
            lambda mid: "A" if str(mid)[-1] in "02468" else "B"
        )

    a = df[df["_sf"] == "A"]
    b = df[df["_sf"] == "B"]

    scale = 1_000_000  # report in ZMW millions

    # Actuarial apportionment weights by benefit type
    W_AGE_RET    = 0.720
    W_ILL_HEALTH = 0.067
    W_WITHDRAWAL = 0.168
    W_DEATH      = 0.045

    def _future_split(sub_df, weight):
        ratio = sub_df["years_to_retirement"] / (
            sub_df["pensionable_service_years"] + sub_df["years_to_retirement"]
        ).replace(0, np.nan)
        fsl = (sub_df["pv_liability"] * ratio.fillna(0.5)).sum()
        return fsl * weight / scale

    def _past_split(sub_df, weight):
        total  = sub_df["pv_liability"].sum() * weight / scale
        future = _future_split(sub_df, weight)
        return total - future

    sal_a = a["annual_pensionable_salary"].sum() / scale
    sal_b = b["annual_pensionable_salary"].sum() / scale
    sal_t = sal_a + sal_b

    fsl = {
        "age_ret_a":    _future_split(a, W_AGE_RET),
        "age_ret_b":    _future_split(b, W_AGE_RET),
        "ill_a":        _future_split(a, W_ILL_HEALTH),
        "ill_b":        _future_split(b, W_ILL_HEALTH),
        "withdrawal_a": _future_split(a, W_WITHDRAWAL),
        "withdrawal_b": _future_split(b, W_WITHDRAWAL),
        "death_a":      _future_split(a, W_DEATH),
        "death_b":      _future_split(b, W_DEATH),
        "sal_a": sal_a,
        "sal_b": sal_b,
    }
    fsl["total_fsl_a"] = fsl["age_ret_a"] + fsl["ill_a"] + fsl["withdrawal_a"] + fsl["death_a"]
    fsl["total_fsl_b"] = fsl["age_ret_b"] + fsl["ill_b"] + fsl["withdrawal_b"] + fsl["death_b"]
    fsl["scr_a"] = fsl["total_fsl_a"] / fsl["sal_a"] if fsl["sal_a"] else 0
    fsl["scr_b"] = fsl["total_fsl_b"] / fsl["sal_b"] if fsl["sal_b"] else 0
    fsl["scr_t"] = (fsl["total_fsl_a"] + fsl["total_fsl_b"]) / sal_t if sal_t else 0
    fsl["total_cr_a"] = fsl["scr_a"] + 0.010
    fsl["total_cr_b"] = fsl["scr_b"] + 0.008
    fsl["total_cr_t"] = fsl["scr_t"] + 0.018
    fsl["ee_rate_a"] = a["employee_contribution_rate"].median() if len(a) > 0 else 0.0725
    fsl["ee_rate_b"] = b["employee_contribution_rate"].median() if len(b) > 0 else 0.0725
    fsl["ee_rate_t"] = df["employee_contribution_rate"].median() if len(df) > 0 else 0.0725
    fsl["boc_a"] = fsl["total_cr_a"] - fsl["ee_rate_a"]
    fsl["boc_b"] = fsl["total_cr_b"] - fsl["ee_rate_b"]
    fsl["boc_t"] = fsl["total_cr_t"] - fsl["ee_rate_t"]

    psl = {
        "age_ret_a":    _past_split(a, W_AGE_RET),
        "age_ret_b":    _past_split(b, W_AGE_RET),
        "ill_a":        _past_split(a, W_ILL_HEALTH),
        "ill_b":        _past_split(b, W_ILL_HEALTH),
        "withdrawal_a": _past_split(a, W_WITHDRAWAL),
        "withdrawal_b": _past_split(b, W_WITHDRAWAL),
        "death_a":      _past_split(a, W_DEATH),
        "death_b":      _past_split(b, W_DEATH),
    }
    psl["total_a"] = psl["age_ret_a"] + psl["ill_a"] + psl["withdrawal_a"] + psl["death_a"]
    psl["total_b"] = psl["age_ret_b"] + psl["ill_b"] + psl["withdrawal_b"] + psl["death_b"]

    return {
        "fsl": fsl, "psl": psl,
        "sal_a": sal_a, "sal_b": sal_b, "sal_t": sal_t,
        "members_a": len(a), "members_b": len(b),
        "scale": scale,
    }


def _fmt_m(v: float, decimals: int = 0) -> str:
    if decimals == 0:
        return f"{v:,.0f}"
    return f"{v:,.{decimals}f}"


def _fmt_rate(v: float) -> str:
    return f"{v:.2%}"


def _table_html(title: str, headers: list, rows: list, totals: list = None,
                note: str = "", accent: str = "") -> str:
    _a = accent or ACCENT_AMBER
    header_cells = "".join(
        f'<th style="padding:0.6rem 1rem; text-align:{"left" if i==0 else "right"}; '
        f'color:{TEXT_SECONDARY}; font-size:0.65rem; letter-spacing:0.12em; '
        f'border-bottom:2px solid {_a}; white-space:nowrap;">{h}</th>'
        for i, h in enumerate(headers)
    )
    def _row_html(cells, is_total=False):
        bg     = f"background:{PANEL_BG};" if is_total else "background:transparent;"
        border = f"border-top:1px solid {_a};" if is_total else f"border-bottom:1px solid {PANEL_BORDER};"
        weight = "font-weight:600;" if is_total else "font-weight:400;"
        col_ov = f"color:{_a};" if is_total else ""
        tds = "".join(
            f'<td style="padding:0.55rem 1rem; text-align:{"left" if i==0 else "right"}; '
            f'font-family:DM Sans,monospace; font-size:0.78rem; '
            f'{border}{bg}{weight}{col_ov if i>0 else ""}">{c}</td>'
            for i, c in enumerate(cells)
        )
        return f"<tr>{tds}</tr>"

    body = "".join(_row_html(r) for r in rows)
    if totals:
        body += "".join(_row_html(r, is_total=True) for r in totals)
    note_html = (
        f'<p style="font-family:DM Sans,monospace;font-size:0.65rem;'
        f'color:{TEXT_SECONDARY};margin-top:0.5rem;">* {note}</p>'
        if note else ""
    )
    return f"""
    <div style="margin-bottom:2rem;">
      <div style="font-family:DM Sans,monospace;font-size:0.65rem;
                  letter-spacing:0.18em;text-transform:uppercase;
                  color:{_a};margin-bottom:0.75rem;">{title}</div>
      <div style="overflow-x:auto;">
        <table style="width:100%;border-collapse:collapse;
                      background:{PANEL_BG};border:1px solid {PANEL_BORDER};
                      font-family:DM Sans,monospace;">
          <thead><tr style="background:{DARK_BG};">{header_cells}</tr></thead>
          <tbody>{body}</tbody>
        </table>
      </div>
      {note_html}
    </div>"""


def _styled_table(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


def render_actuarial_output(engine: "PensionEngine") -> None:
    section("ACTUARIAL VALUATION OUTPUT")

    st.markdown(
        f'<div class="info-banner">Values are in ZMW millions (ZMW\u2019m). '
        f'Sub Fund A / B split is derived from member_id parity — '
        f'add a <code>sub_fund</code> column (A or B) to your CSV to use real fund codes.</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Computing actuarial output tables\u2026"):
        out = _compute_output_tables(engine)

    fsl = out["fsl"]
    psl = out["psl"]

    # KPI strip
    k1, k2, k3, k4 = st.columns(4)
    total_fsl = fsl["total_fsl_a"] + fsl["total_fsl_b"]
    total_psl = psl["total_a"]     + psl["total_b"]
    with k1:
        st.markdown(kpi_html("MEMBERS — SUB FUND A", f'{out["members_a"]:,}',
            f'Sal. Bill: ZMW {_fmt_m(out["sal_a"])}M', "info"), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_html("MEMBERS — SUB FUND B", f'{out["members_b"]:,}',
            f'Sal. Bill: ZMW {_fmt_m(out["sal_b"])}M', "info"), unsafe_allow_html=True)
    with k3:
        st.markdown(kpi_html("FUTURE SERVICE LIABILITY", f'ZMW {_fmt_m(total_fsl)}M',
            "Combined FSL", "alert"), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_html("PAST SERVICE LIABILITY", f'ZMW {_fmt_m(total_psl)}M',
            "Combined PSL", "alert"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    def _t(a_val, b_val):
        return (_fmt_m(a_val), _fmt_m(b_val), _fmt_m(a_val + b_val))

    # ── TABLE 1: Future Service Liability ─────────────────────────────────
    _styled_table(_table_html(
        "TABLE 1 — FUTURE SERVICE LIABILITY",
        ["Future Service Liability", "Sub Fund A (ZMW\u2019m)", "Sub Fund B (ZMW\u2019m)", "Total (ZMW\u2019m)"],
        [
            ("Age Retirement Benefits",         *_t(fsl["age_ret_a"],    fsl["age_ret_b"])),
            ("Ill Health Retirement Benefits",  *_t(fsl["ill_a"],        fsl["ill_b"])),
            ("Refund on Withdrawal",            *_t(fsl["withdrawal_a"], fsl["withdrawal_b"])),
            ("Death Benefit",                   *_t(fsl["death_a"],      fsl["death_b"])),
        ],
        totals=[
            ("Value of future service liability",
             _fmt_m(fsl["total_fsl_a"]), _fmt_m(fsl["total_fsl_b"]),
             _fmt_m(fsl["total_fsl_a"] + fsl["total_fsl_b"])),
            ("Present value of future salary bill over entire future working life",
             _fmt_m(fsl["sal_a"]), _fmt_m(fsl["sal_b"]),
             _fmt_m(fsl["sal_a"] + fsl["sal_b"])),
        ],
        note="Includes cost of administration expenses and risk benefit premiums",
    ))

    # ── TABLE 1B: Standard Contribution Rate ──────────────────────────────
    _styled_table(_table_html(
        "TABLE 1B — STANDARD CONTRIBUTION RATE DERIVATION",
        ["Contribution Rate Analysis", "Sub Fund A", "Sub Fund B", "Total"],
        [
            ("Standard Contribution rate",
             _fmt_rate(fsl["scr_a"]), _fmt_rate(fsl["scr_b"]), _fmt_rate(fsl["scr_t"])),
            ("Add: Cost of other benefits* and admin expenses",
             _fmt_rate(0.010), _fmt_rate(0.008), _fmt_rate(0.018)),
            ("Total Contribution rate",
             _fmt_rate(fsl["total_cr_a"]), _fmt_rate(fsl["total_cr_b"]), _fmt_rate(fsl["total_cr_t"])),
            ("Less: Employee Contribution rate",
             _fmt_rate(fsl["ee_rate_a"]), _fmt_rate(fsl["ee_rate_b"]), _fmt_rate(fsl["ee_rate_t"])),
        ],
        totals=[
            ("Balance of Cost",
             _fmt_rate(fsl["boc_a"]), _fmt_rate(fsl["boc_b"]), _fmt_rate(fsl["boc_t"])),
        ],
        note="Admin expenses loading: Sub Fund A 1.00%, Sub Fund B 0.80%",
        accent=ACCENT_BLUE,
    ))

    # ── TABLE 2: Past Service Liability ───────────────────────────────────
    _styled_table(_table_html(
        "TABLE 2 — PAST SERVICE LIABILITY",
        ["Past Service Liability", "Sub Fund A (ZMW\u2019m)", "Sub Fund B (ZMW\u2019m)", "Total (ZMW\u2019m)"],
        [
            ("Age Retirement Benefits",         *_t(psl["age_ret_a"],    psl["age_ret_b"])),
            ("Ill Health Retirement Benefits",  *_t(psl["ill_a"],        psl["ill_b"])),
            ("Refund on Withdrawal",            *_t(psl["withdrawal_a"], psl["withdrawal_b"])),
            ("Death Benefit",                   *_t(psl["death_a"],      psl["death_b"])),
        ],
        totals=[
            ("Total",
             _fmt_m(psl["total_a"]), _fmt_m(psl["total_b"]),
             _fmt_m(psl["total_a"] + psl["total_b"])),
        ],
        accent=ACCENT_GREEN,
    ))

    # ── TABLE 3: Reconciliation ───────────────────────────────────────────
    pm             = engine.portfolio_metrics
    total_liability = total_fsl + total_psl
    total_assets   = pm.total_accumulated / out["scale"]
    surplus        = total_assets - total_liability
    funding        = total_assets / total_liability if total_liability else 0

    _styled_table(_table_html(
        "TABLE 3 — TOTAL LIABILITY RECONCILIATION",
        ["Component", "ZMW\u2019m", "% of Total"],
        [
            ("Future Service Liability",
             _fmt_m(total_fsl),
             f'{total_fsl/total_liability:.1%}' if total_liability else "\u2014"),
            ("Past Service Liability",
             _fmt_m(total_psl),
             f'{total_psl/total_liability:.1%}' if total_liability else "\u2014"),
            ("Total Accumulated Fund (Assets)",
             _fmt_m(total_assets), "\u2014"),
        ],
        totals=[
            ("Total Actuarial Liability", _fmt_m(total_liability), "100.0%"),
            ("Surplus / (Deficit)",
             _fmt_m(surplus),
             f'Funding Ratio: {funding:.1%}'),
        ],
    ))

    # ── Visual breakdown ──────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section("VISUAL BREAKDOWN")

    c1, c2 = st.columns(2)
    with c1:
        benefit_labels = ["Age Retirement", "Ill Health", "Withdrawal", "Death"]
        fsl_vals = [fsl["age_ret_a"]+fsl["age_ret_b"], fsl["ill_a"]+fsl["ill_b"],
                    fsl["withdrawal_a"]+fsl["withdrawal_b"], fsl["death_a"]+fsl["death_b"]]
        psl_vals = [psl["age_ret_a"]+psl["age_ret_b"], psl["ill_a"]+psl["ill_b"],
                    psl["withdrawal_a"]+psl["withdrawal_b"], psl["death_a"]+psl["death_b"]]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Future Service", x=benefit_labels, y=fsl_vals,
                             marker_color=ACCENT_AMBER, opacity=0.9))
        fig.add_trace(go.Bar(name="Past Service",   x=benefit_labels, y=psl_vals,
                             marker_color=ACCENT_BLUE, opacity=0.9))
        fig.update_layout(barmode="group", title="FSL vs PSL by Benefit Type (ZMW\u2019m)",
                          yaxis_title="ZMW Millions", height=320,
                          legend=dict(orientation="h", y=1.1))
        apply_template(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        cats  = ["Sub Fund A", "Sub Fund B", "Total"]
        fig2  = go.Figure()
        fig2.add_trace(go.Bar(name="Standard CR (%)", x=cats,
                              y=[fsl["scr_a"]*100, fsl["scr_b"]*100, fsl["scr_t"]*100],
                              marker_color=ACCENT_AMBER, opacity=0.85))
        fig2.add_trace(go.Bar(name="Total CR incl. admin (%)", x=cats,
                              y=[fsl["total_cr_a"]*100, fsl["total_cr_b"]*100, fsl["total_cr_t"]*100],
                              marker_color=ACCENT_BLUE, opacity=0.65))
        fig2.add_trace(go.Scatter(name="Balance of Cost — Employer (%)", x=cats,
                                  y=[fsl["boc_a"]*100, fsl["boc_b"]*100, fsl["boc_t"]*100],
                                  mode="lines+markers+text",
                                  line=dict(color=ACCENT_GREEN, width=2),
                                  marker=dict(size=10),
                                  text=[f'{v*100:.2f}%' for v in [fsl["boc_a"], fsl["boc_b"], fsl["boc_t"]]],
                                  textposition="top center", textfont=dict(size=10)))
        fig2.update_layout(barmode="overlay", title="Contribution Rate Analysis (%)",
                           yaxis_title="Rate (%)", height=320,
                           legend=dict(orientation="h", y=1.1))
        apply_template(fig2)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    c3, c4 = st.columns(2)
    with c3:
        fig3 = go.Figure(go.Pie(
            labels=["Sub Fund A — FSL", "Sub Fund B — FSL",
                    "Sub Fund A — PSL", "Sub Fund B — PSL"],
            values=[fsl["total_fsl_a"], fsl["total_fsl_b"],
                    psl["total_a"],     psl["total_b"]],
            hole=0.55,
            marker=dict(colors=[ACCENT_AMBER, ACCENT_BLUE, "#fb923c", "#a78bfa"]),
            textinfo="label+percent",
            textfont=dict(size=10, family="DM Sans"),
        ))
        fig3.update_layout(title="Liability Split — Sub Fund \u00d7 Service Component",
                           height=320, showlegend=False)
        apply_template(fig3)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with c4:
        fig4 = go.Figure()
        fig4.add_trace(go.Bar(name="Employee Rate (%)", x=cats,
                              y=[fsl["ee_rate_a"]*100, fsl["ee_rate_b"]*100, fsl["ee_rate_t"]*100],
                              marker_color=ACCENT_SLATE, opacity=0.8))
        fig4.add_trace(go.Bar(name="Balance of Cost — Employer (%)", x=cats,
                              y=[fsl["boc_a"]*100, fsl["boc_b"]*100, fsl["boc_t"]*100],
                              marker_color=ACCENT_GREEN, opacity=0.85))
        fig4.update_layout(barmode="stack",
                           title="Contribution Split: Employee vs Employer (%)",
                           yaxis_title="Contribution Rate (%)", height=320,
                           legend=dict(orientation="h", y=1.1))
        apply_template(fig4)
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})



# ---------------------------------------------------------------------------
# Solvency Position Page
# ---------------------------------------------------------------------------

@st.cache_data(ttl=600, show_spinner=False)
def _load_financials() -> "FundFinancials":
    return load_fund_financials()


def render_solvency_position(engine: "PensionEngine") -> None:
    section("SOLVENCY POSITION")

    st.markdown(
        f'<div class="info-banner">Solvency position combines actuarial liabilities '
        f'(FSL + PSL from the Actuarial Output page) with the fund&#8217;s audited net assets '
        f'from the Balance Sheet. Assets are shown at adjusted net asset value per the '
        f'Revenue Account financial statements.</div>',
        unsafe_allow_html=True,
    )

    ff = _load_financials()
    with st.spinner("Computing solvency position…"):
        out   = _compute_output_tables(engine)
        fsl   = out["fsl"]
        psl   = out["psl"]
        sp    = compute_solvency_position(
            fsl_a               = fsl["total_fsl_a"],
            fsl_b               = fsl["total_fsl_b"],
            psl_a               = psl["total_a"],
            psl_b               = psl["total_b"],
            total_pv_liability  = engine.portfolio_metrics.total_pv_liability,
            financials          = ff,
            valuation_year      = 2023,
        )

    # ── KPI strip ──────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    sol_variant = "success" if sp.solvency_ratio >= 1.0 else ("" if sp.solvency_ratio >= 0.8 else "alert")
    with k1:
        st.markdown(kpi_html("TOTAL FUND ASSETS", f"ZMW {_fmt_m(sp.total_assets)}M",
            "Adjusted net assets", "info"), unsafe_allow_html=True)
    with k2:
        st.markdown(kpi_html("TOTAL LIABILITY (FSL+PSL)", f"ZMW {_fmt_m(sp.total_liability)}M",
            "FSL + PSL combined", "alert"), unsafe_allow_html=True)
    with k3:
        surplus_label = "SURPLUS" if sp.surplus_deficit >= 0 else "DEFICIT"
        surplus_var   = "success" if sp.surplus_deficit >= 0 else "alert"
        st.markdown(kpi_html(surplus_label, f"ZMW {_fmt_m(abs(sp.surplus_deficit))}M",
            "Assets minus Liabilities", surplus_var), unsafe_allow_html=True)
    with k4:
        st.markdown(kpi_html("SOLVENCY RATIO", f"{sp.solvency_ratio:.1%}",
            "Assets / Total Liability", sol_variant), unsafe_allow_html=True)
    with k5:
        st.markdown(kpi_html("FUNDING LEVEL", f"{sp.funding_level_pct:.1f}%",
            "100% = fully funded", sol_variant), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Main solvency table ────────────────────────────────────────────────
    def _tr(a_val, b_val):
        return (_fmt_m(a_val), _fmt_m(b_val), _fmt_m(a_val + b_val))

    _styled_table(_table_html(
        "TABLE — SOLVENCY POSITION",
        ["", "Sub Fund A (ZMW\u2019m)", "Sub Fund B (ZMW\u2019m)", "Total (ZMW\u2019m)"],
        [
            ("FUTURE SERVICE LIABILITY", "", "", ""),
            ("  Age Retirement Benefits",         *_tr(fsl["age_ret_a"],    fsl["age_ret_b"])),
            ("  Ill Health Retirement Benefits",  *_tr(fsl["ill_a"],        fsl["ill_b"])),
            ("  Refund on Withdrawal",            *_tr(fsl["withdrawal_a"], fsl["withdrawal_b"])),
            ("  Death Benefit",                   *_tr(fsl["death_a"],      fsl["death_b"])),
            ("Total FSL",
             _fmt_m(fsl["total_fsl_a"]), _fmt_m(fsl["total_fsl_b"]),
             _fmt_m(fsl["total_fsl_a"] + fsl["total_fsl_b"])),
            ("PAST SERVICE LIABILITY", "", "", ""),
            ("  Age Retirement Benefits",         *_tr(psl["age_ret_a"],    psl["age_ret_b"])),
            ("  Ill Health Retirement Benefits",  *_tr(psl["ill_a"],        psl["ill_b"])),
            ("  Refund on Withdrawal",            *_tr(psl["withdrawal_a"], psl["withdrawal_b"])),
            ("  Death Benefit",                   *_tr(psl["death_a"],      psl["death_b"])),
            ("Total PSL",
             _fmt_m(psl["total_a"]), _fmt_m(psl["total_b"]),
             _fmt_m(psl["total_a"] + psl["total_b"])),
        ],
        totals=[
            ("Total Actuarial Liability (FSL + PSL)",
             _fmt_m(fsl["total_fsl_a"] + psl["total_a"]),
             _fmt_m(fsl["total_fsl_b"] + psl["total_b"]),
             _fmt_m(sp.total_liability)),
            ("Fund Assets (Adjusted Net Assets)",
             _fmt_m(sp.assets_a), _fmt_m(sp.assets_b),
             _fmt_m(sp.total_assets)),
            ("Surplus / (Deficit)",
             _fmt_m(sp.assets_a - (fsl["total_fsl_a"] + psl["total_a"])),
             _fmt_m(sp.assets_b - (fsl["total_fsl_b"] + psl["total_b"])),
             _fmt_m(sp.surplus_deficit)),
            ("Solvency Ratio",
             f"{sp.solvency_ratio_a:.1%}", f"{sp.solvency_ratio_b:.1%}",
             f"{sp.solvency_ratio:.1%}"),
        ],
    ))

    # ── Financial history table ────────────────────────────────────────────
    if ff.balance_sheets:
        _styled_table(_table_html(
            "FUND BALANCE SHEET SUMMARY — HISTORICAL",
            ["", "2020", "2021", "2022", "2023"],
            [
                ("Total Assets (ZMW)",) + tuple(
                    f"{(ff.balance_sheet(yr).total_assets or 0):,.0f}"
                    for yr in [2020,2021,2022,2023]
                ),
                ("Total Liabilities (ZMW)",) + tuple(
                    f"{(ff.balance_sheet(yr).total_liabilities or 0):,.0f}"
                    for yr in [2020,2021,2022,2023]
                ),
                ("Net Assets (ZMW)",) + tuple(
                    f"{(ff.balance_sheet(yr).net_assets or 0):,.0f}"
                    for yr in [2020,2021,2022,2023]
                ),
                ("Government Securities",) + tuple(
                    f"{(ff.balance_sheet(yr).government_securities or 0):,.0f}"
                    for yr in [2020,2021,2022,2023]
                ),
                ("Equity Investments",) + tuple(
                    f"{(ff.balance_sheet(yr).equity_investments or 0):,.0f}"
                    for yr in [2020,2021,2022,2023]
                ),
                ("Cash at Bank",) + tuple(
                    f"{(ff.balance_sheet(yr).cash_at_bank or 0):,.0f}"
                    for yr in [2020,2021,2022,2023]
                ),
            ],
            accent=ACCENT_BLUE,
        ))

    # ── Income statement history table ────────────────────────────────────
    if ff.income_statements:
        _styled_table(_table_html(
            "INCOME STATEMENT SUMMARY — HISTORICAL",
            ["", "2021", "2022", "2023"],
            [
                ("Total Income (ZMW)",) + tuple(
                    f"{(ff.income_statement(yr).total_income or 0):,.0f}"
                    for yr in ["2021","2022","2023"]
                ),
                ("Total Expenditure (ZMW)",) + tuple(
                    f"{(ff.income_statement(yr).total_expenditure or 0):,.0f}"
                    for yr in ["2021","2022","2023"]
                ),
                ("Employee Contributions",) + tuple(
                    f"{(ff.income_statement(yr).employee_contributions or 0):,.0f}"
                    for yr in ["2021","2022","2023"]
                ),
                ("Employer Contributions",) + tuple(
                    f"{(ff.income_statement(yr).employer_contributions or 0):,.0f}"
                    for yr in ["2021","2022","2023"]
                ),
                ("Investment Income",) + tuple(
                    f"{(ff.income_statement(yr).investment_income or 0):,.0f}"
                    for yr in ["2021","2022","2023"]
                ),
                ("Pensions Paid",) + tuple(
                    f"{(ff.income_statement(yr).pensions_paid or 0):,.0f}"
                    for yr in ["2021","2022","2023"]
                ),
                ("Surplus",) + tuple(
                    f"{(ff.income_statement(yr).surplus or 0):,.0f}"
                    for yr in ["2021","2022","2023"]
                ),
            ],
            accent=ACCENT_GREEN,
        ))

    # ── Visualisations ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    section("SOLVENCY VISUALISATIONS")

    c1, c2 = st.columns(2)

    with c1:
        # Assets vs liability waterfall
        fig = go.Figure(go.Bar(
            x=["Sub Fund A", "Sub Fund B", "Total"],
            y=[sp.assets_a, sp.assets_b, sp.total_assets],
            name="Assets", marker_color=ACCENT_GREEN, opacity=0.85,
        ))
        fig.add_trace(go.Bar(
            x=["Sub Fund A", "Sub Fund B", "Total"],
            y=[fsl["total_fsl_a"]+psl["total_a"],
               fsl["total_fsl_b"]+psl["total_b"],
               sp.total_liability],
            name="Total Liability", marker_color=ACCENT_RED, opacity=0.8,
        ))
        fig.update_layout(
            barmode="group", title="Assets vs Total Liability by Sub Fund (ZMW\u2019m)",
            yaxis_title="ZMW Millions", height=340,
            legend=dict(orientation="h", y=1.1),
        )
        apply_template(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        # Net assets trend
        years  = sorted([bs.year for bs in ff.balance_sheets])
        assets = [ff.balance_sheet(y).net_assets for y in years]
        fig2   = go.Figure()
        fig2.add_trace(go.Scatter(
            x=years, y=assets, mode="lines+markers+text",
            line=dict(color=ACCENT_BLUE, width=3),
            marker=dict(size=9, color=ACCENT_BLUE),
            fill="tozeroy", fillcolor=f"{ACCENT_BLUE}18",
            text=[f"ZMW {v/1e9:.1f}B" for v in assets],
            textposition="top center", textfont=dict(size=9),
        ))
        fig2.update_layout(
            title="Net Assets Trend (ZMW)", yaxis_title="ZMW", height=340,
            xaxis=dict(tickvals=years, ticktext=[str(y) for y in years]),
        )
        apply_template(fig2)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    c3, c4 = st.columns(2)
    with c3:
        # Asset composition latest year
        bs_latest = ff.latest_balance_sheet()
        if bs_latest:
            asset_items = {
                "Govt Securities":   bs_latest.government_securities,
                "Fixed Deposits":    bs_latest.fixed_deposits,
                "Equity":            bs_latest.equity_investments,
                "Investment Prop":   bs_latest.investment_properties,
                "Microfinance":      bs_latest.microfinance_loans,
                "Home Loans":        bs_latest.home_loan_scheme,
                "WIP":               bs_latest.work_in_progress,
                "Cash & Other":      bs_latest.cash_at_bank + bs_latest.other_receivables,
            }
            labels = list(asset_items.keys())
            values = [max(v, 0) for v in asset_items.values()]
            colors = [ACCENT_BLUE, ACCENT_AMBER, ACCENT_GREEN, "#7C3AED",
                      "#0369A1", "#F59E0B", ACCENT_SLATE, "#E5E7EB"]
            fig3 = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.55,
                marker=dict(colors=colors),
                textinfo="label+percent", textfont=dict(size=9, family="DM Sans"),
            ))
            fig3.update_layout(
                title=f"Asset Composition — {bs_latest.year}",
                height=340, showlegend=False,
            )
            apply_template(fig3)
            st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

    with c4:
        # Income vs expenditure bar
        if ff.income_statements:
            yrs  = [is_.year_end for is_ in ff.income_statements]
            inc  = [is_.total_income for is_ in ff.income_statements]
            exp  = [is_.total_expenditure for is_ in ff.income_statements]
            sur  = [is_.surplus for is_ in ff.income_statements]
            fig4 = go.Figure()
            fig4.add_trace(go.Bar(name="Total Income",      x=yrs, y=inc, marker_color=ACCENT_GREEN, opacity=0.85))
            fig4.add_trace(go.Bar(name="Total Expenditure", x=yrs, y=exp, marker_color=ACCENT_RED,   opacity=0.75))
            fig4.add_trace(go.Scatter(name="Surplus", x=yrs, y=sur,
                mode="lines+markers", line=dict(color=ACCENT_BLUE, width=2),
                marker=dict(size=9)))
            fig4.update_layout(
                barmode="group", title="Income vs Expenditure (ZMW)",
                yaxis_title="ZMW", height=340,
                legend=dict(orientation="h", y=1.1),
            )
            apply_template(fig4)
            st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Actuarial Tables Reference Page
# ---------------------------------------------------------------------------

def render_actuarial_tables_page() -> None:
    section("ACTUARIAL TABLES REFERENCE")

    st.markdown(
        f'<div class="info-banner">Live actuarial tables loaded from uploaded files. '
        f'Mortality: a(55) immediate annuitant table (male/female). '
        f'Ill-health: CMI WP50 ACMNL04/ACFNL04 critical illness rates (non-smoker).</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs([
        "  a(55) Mortality Table  ",
        "  CMI WP50 Ill-Health Rates  ",
        "  Annuity Factors  ",
    ])

    with tab1:
        st.markdown(
            f'<p style="font-family:\'DM Sans\',sans-serif; font-size:0.78rem; '
            f'color:{TEXT_SECONDARY}; margin-bottom:1rem;">'
            f'One-year select death probabilities q[x]. Duration 0 = first year of annuity; '
            f'Duration 1+ = ultimate. Source: a(55) Immediate Annuitants table.</p>',
            unsafe_allow_html=True,
        )
        df_mort = mortality_table_summary()
        st.dataframe(df_mort, use_container_width=True, hide_index=True)

        # Visual
        t = get_mortality_table()
        ages_plot = list(range(20, 101, 1))
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=ages_plot,
            y=[t.qx(a, "Male", 1) for a in ages_plot],
            name="Male q[x]", mode="lines",
            line=dict(color=ACCENT_BLUE, width=2),
        ))
        fig.add_trace(go.Scatter(
            x=ages_plot,
            y=[t.qx(a, "Female", 1) for a in ages_plot],
            name="Female q[x]", mode="lines",
            line=dict(color=ACCENT_AMBER, width=2),
        ))
        fig.update_layout(
            title="a(55) Mortality Rates q[x] — Duration 1+ (Ultimate)",
            xaxis_title="Age", yaxis_title="q[x]",
            height=350, legend=dict(orientation="h", y=1.1),
        )
        apply_template(fig)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with tab2:
        st.markdown(
            f'<p style="font-family:\'DM Sans\',sans-serif; font-size:0.78rem; '
            f'color:{TEXT_SECONDARY}; margin-bottom:1rem;">'
            f'CMI WP50 critical illness / ill-health inception rates. '
            f'ACMNL04 = Male Non-Smoker; ACFNL04 = Female Non-Smoker. '
            f'Duration 5+ (ultimate) rates used in benefit calculations.</p>',
            unsafe_allow_html=True,
        )
        df_ih = ill_health_table_summary()
        st.dataframe(df_ih, use_container_width=True, hide_index=True)

        ih = get_ill_health_table()
        ages_ih = list(range(18, 66))
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=ages_ih,
            y=[ih.qx(a, "Male", 5) for a in ages_ih],
            name="Male Dur 5+", mode="lines",
            line=dict(color=ACCENT_BLUE, width=2),
        ))
        fig2.add_trace(go.Scatter(
            x=ages_ih,
            y=[ih.qx(a, "Female", 5) for a in ages_ih],
            name="Female Dur 5+", mode="lines",
            line=dict(color=ACCENT_AMBER, width=2),
        ))
        fig2.add_trace(go.Scatter(
            x=ages_ih,
            y=[ih.qx(a, "Male", 0) for a in ages_ih],
            name="Male Dur 0", mode="lines",
            line=dict(color=ACCENT_BLUE, width=1.5, dash="dot"),
        ))
        fig2.update_layout(
            title="CMI WP50 Ill-Health Rates q[x] by Duration",
            xaxis_title="Age Exact", yaxis_title="q[x]",
            height=350, legend=dict(orientation="h", y=1.1),
        )
        apply_template(fig2)
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with tab3:
        st.markdown(
            f'<p style="font-family:\'DM Sans\',sans-serif; font-size:0.78rem; '
            f'color:{TEXT_SECONDARY}; margin-bottom:1rem;">'
            f'Whole-life annuity factors ä_x computed from a(55) mortality at various '
            f'ages and discount rates. Used to discount projected pension benefits '
            f'to present value.</p>',
            unsafe_allow_html=True,
        )
        df_af = annuity_factor_grid()
        st.dataframe(df_af, use_container_width=True, hide_index=True)

        t2 = get_mortality_table()
        ages_af = list(range(20, 71, 1))
        fig3 = go.Figure()
        for rate, colour in [(0.05, "#0369A1"), (0.08, ACCENT_BLUE),
                              (0.10, ACCENT_AMBER), (0.12, ACCENT_RED)]:
            fig3.add_trace(go.Scatter(
                x=ages_af,
                y=[t2.annuity_factor(a, "Male", rate) for a in ages_af],
                name=f"Male {rate:.0%}", mode="lines",
                line=dict(color=colour, width=2),
            ))
        fig3.update_layout(
            title="Whole-Life Annuity Factors ä_x (Male) — Various Discount Rates",
            xaxis_title="Age at Retirement", yaxis_title="ä_x",
            height=350, legend=dict(orientation="h", y=1.1),
        )
        apply_template(fig3)
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(engine: PensionEngine) -> str:
    # Brand lockup
    st.sidebar.markdown(
        f'<div style="padding:1.5rem 1rem 1rem 1rem; border-bottom:1px solid rgba(255,255,255,0.1); margin-bottom:1rem;">'
        f'<div style="font-family:\'DM Serif Display\',serif; font-size:1.1rem; '
        f'color:#FFFFFF; font-weight:400; letter-spacing:-0.01em;">Pension Intelligence</div>'
        f'<div style="font-family:\'DM Sans\',sans-serif; font-size:0.6rem; '
        f'font-weight:700; letter-spacing:0.14em; text-transform:uppercase; '
        f'color:{SIDEBAR_ACCENT}; margin-top:0.2rem;">Actuarial Platform</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f'<p style="font-family:\'DM Sans\',sans-serif; font-size:0.6rem; '
        f'font-weight:700; letter-spacing:0.14em; text-transform:uppercase; '
        f'color:rgba(255,255,255,0.4); padding:0 1rem 0.4rem 1rem;">Navigation</p>',
        unsafe_allow_html=True,
    )

    page = st.sidebar.radio(
        "",
        [
            "◈ Executive Overview",
            "⬡ Liability Analysis",
            "◎ Contribution Analysis",
            "△ Scenario Testing",
            "⚠ Risk Detection",
            "≡ Member Explorer",
            "⊞ Actuarial Output",
            "◇ Solvency Position",
            "⬒ Actuarial Tables",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.markdown(f'<hr style="border-color:{PANEL_BORDER};">', unsafe_allow_html=True)

    pm = engine.portfolio_metrics
    st.sidebar.markdown(
        f"""
        <div style="font-family:'DM Sans',sans-serif; font-size:0.72rem;
                    padding:0.75rem 1rem; margin:0.5rem 0;
                    background:rgba(255,255,255,0.05); border-radius:8px;
                    border:1px solid rgba(255,255,255,0.08);">
          <div style="font-size:0.58rem; font-weight:700; letter-spacing:0.14em;
                      text-transform:uppercase; color:{SIDEBAR_ACCENT};
                      margin-bottom:0.6rem;">Scheme Vitals</div>
          <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
            <span style="color:rgba(255,255,255,0.55);">Members</span>
            <span style="color:#FFFFFF; font-weight:500;">{pm.total_members:,}</span>
          </div>
          <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
            <span style="color:rgba(255,255,255,0.55);">Fund</span>
            <span style="color:#FFFFFF; font-weight:500;">{fmt_currency(pm.total_accumulated)}</span>
          </div>
          <div style="display:flex; justify-content:space-between; margin-bottom:0.3rem;">
            <span style="color:rgba(255,255,255,0.55);">Liability</span>
            <span style="color:#FFFFFF; font-weight:500;">{fmt_currency(pm.total_pv_liability)}</span>
          </div>
          <div style="height:1px; background:rgba(255,255,255,0.08); margin:0.5rem 0;"></div>
          <div style="display:flex; justify-content:space-between;">
            <span style="color:rgba(255,255,255,0.55);">Funding</span>
            <span style="color:{'#86BC25' if pm.funding_ratio >= 1 else '#F87171'}; font-weight:600;">
              {fmt_pct(pm.funding_ratio)}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        f'<div style="padding:1rem; margin-top:1rem; border-top:1px solid rgba(255,255,255,0.08);">'
        f'<div style="font-family:\'DM Sans\',sans-serif; font-size:0.62rem; '
        f'color:rgba(255,255,255,0.35); line-height:1.6;">'
        f'Pension Actuarial Engine<br>v1.0 · Consulting Grade</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    return page


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)

    engine = get_engine()
    pm     = engine.portfolio_metrics

    render_header()
    page = render_sidebar(engine)

    if page == "◈ Executive Overview":
        render_kpi_row(pm)
        st.markdown("<br>", unsafe_allow_html=True)
        render_secondary_kpis(pm)
        st.markdown("<br>", unsafe_allow_html=True)
        render_overview_charts(engine, pm)

    elif page == "⬡ Liability Analysis":
        render_liability_analysis(engine)

    elif page == "◎ Contribution Analysis":
        render_kpi_row(pm)
        st.markdown("<br>", unsafe_allow_html=True)
        render_contribution_analysis(engine)

    elif page == "△ Scenario Testing":
        render_scenario_analysis(engine)

    elif page == "⚠ Risk Detection":
        render_risk_analysis(engine)

    elif page == "≡ Member Explorer":
        render_member_explorer(engine)

    elif page == "⊞ Actuarial Output":
        render_actuarial_output(engine)

    elif page == "◇ Solvency Position":
        render_solvency_position(engine)

    elif page == "⬒ Actuarial Tables":
        render_actuarial_tables_page()


if __name__ == "__main__":
    main()

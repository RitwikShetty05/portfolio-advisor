"""
app.py
======

Phase 7 — Streamlit frontend for the AI Portfolio Advisory System.

Five pages on the sidebar:

1.  **Dashboard**          — broad-market regime widget, top BUY signals,
                              fixed signal heat-map (real dates, discrete
                              colors, rich hover).
2.  **Stock Analyzer**     — candles + MAs, volume, RSI, MACD, signals,
                              trade plan card.
3.  **Portfolio Analyzer** — upload (CSV/XLSX/PDF) **or** type holdings →
                              risk metrics, correlation heat-map, sector
                              **treemap**, VaR.
4.  **Backtest Lab**       — equity vs NIFTY 50 with max-DD shaded,
                              drawdown, rolling Sharpe, **monthly-returns
                              heat-map**, **returns histogram**,
                              **per-trade P&L bars**, full trade log.
5.  **Recommendations**    — ranked Short-Term / Long-Term / EXIT cards;
                              same upload widget as the Portfolio page.

Design notes
------------
* **Color theme** — restrained Bloomberg-inspired palette built from the
  Okabe-Ito colour-blind-safe set (teal/grey/orange for regimes). One
  global ``THEME`` dict so every chart pulls from the same source.
* **Heat-map redesign** — discrete 3-step colorscale (no gradient
  ambiguity), real date ticks on the x-axis, hover shows ticker + date +
  signal strength + confidence.
* **Caching** is via ``@st.cache_data``. The full pipeline runs once per
  (universe, dates, regime_method) tuple; subsequent navigation is
  instant.
* **Layout density** — collapsible expanders for less-frequently used
  sections (Pipeline settings, QA report, Warnings details). Side-by-side
  columns wherever a chart pairs with a metric or a table.
"""

from __future__ import annotations

import logging
import sys
from datetime import date
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import streamlit as st
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
except ImportError as e:  # pragma: no cover
    raise ImportError("streamlit + plotly required: pip install streamlit plotly") from e

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config as C  # noqa: E402
from src.data_loader import DataLoader  # noqa: E402
from src.features import FeatureEngineer  # noqa: E402
from src.regime import RegimeDetector  # noqa: E402
from src.signals import SignalEngine, add_entry_exit_levels  # noqa: E402
from src.backtest import Backtester  # noqa: E402
from src.portfolio import PortfolioAnalyzer  # noqa: E402
from src.recommend import RecommendationEngine  # noqa: E402
from src.portfolio_parser import parse_holdings_file  # noqa: E402
from src.significance import full_significance_report  # noqa: E402
from src.walkforward import WalkForward  # noqa: E402
from src.factor_attribution import (  # noqa: E402
    FactorAttribution, INDIAN_FACTOR_PROXIES,
)
from src.live_quotes import (  # noqa: E402
    LiveQuote, clear_cache as clear_live_cache, get_live_quote, get_live_quotes,
    is_market_open, market_status_text, now_ist,
)
from src.dynamic_universe import (  # noqa: E402
    NSE_STOCK_NAMES, lazy_load_tickers, normalise_ticker, search_nse_stocks,
)

logging.basicConfig(level=C.LOG_LEVEL, format=C.LOG_FORMAT)
logger = logging.getLogger("app")


# ---------------------------------------------------------------------------
# Streamlit page config — must be the first Streamlit call.
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Portfolio Advisor — NSE",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Global CSS — finance-grade polish
# ---------------------------------------------------------------------------
# Why each rule exists:
#   * `tabular-nums` makes digits the same width, so columns of numbers
#     in tables/metrics align vertically. EVERY trading screen does this.
#     Without it, "23,897" and "11,235" don't line up — a tell-tale sign
#     of "built by someone who doesn't read tables for a living."
#   * `tnum` is the equivalent OpenType feature flag — belt and braces.
#   * Slightly tighter `.block-container` padding because finance UIs are
#     traditionally information-dense (Bloomberg, Reuters, IBKR TWS).
#   * `.stMetric` value gets a font-weight bump — primary KPIs should be
#     visually heavy.
st.markdown(
    """
    <style>
        html, body, [class*="css"], .stMetric, .stDataFrame,
        .stTabs, .stMarkdown, .stPlotlyChart {
            font-variant-numeric: tabular-nums;
            font-feature-settings: "tnum" 1;
        }
        /* Push our content below Streamlit's built-in toolbar (Deploy / menu).
           Without enough top padding the status bar gets clipped. */
        .block-container {
            padding-top: 3.2rem;
            padding-bottom: 2rem;
        }
        [data-testid="stMetricValue"] {
            font-weight: 700;
            letter-spacing: -0.01em;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.78rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        /* Slim down the default radio for the picker rows */
        .stRadio > div { gap: 0.4rem; }
        /* Tighten the segmented_control above the Stock Analyzer chart
           so the chips feel like part of the chart, not a separate widget. */
        div[data-testid="stSegmentedControl"] {
            margin-top: -0.4rem;
            margin-bottom: -0.2rem;
        }
        div[data-testid="stSegmentedControl"] label {
            font-size: 0.78rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Global theme — built from the Okabe-Ito color-blind-safe palette so the
# entire app uses one consistent visual language.
#
# Why these choices:
#   * Teal (BULL) and orange (BEAR) are distinguishable for both common
#     types of color-blindness, unlike pure red/green.
#   * Navy is the institutional fintech colour; we use it for the primary
#     "strategy" line vs a soft grey for the benchmark — readable at a
#     glance without being noisy.
#   * Background-friendly hues: muted enough to avoid eye-strain on
#     large charts but saturated enough to read on a beamer.
# ---------------------------------------------------------------------------
THEME = {
    "bull":          "#0fb5ae",   # teal
    "bear":          "#e15759",   # salmon-red
    "sideways":      "#9ca3af",   # warm grey
    "primary":       "#1e3a5f",   # deep navy — strategy line
    "secondary":     "#f5a623",   # amber — accents / signal markers
    "benchmark":     "#94a3b8",   # cool grey — NIFTY 50 reference
    "grid":          "#e5e7eb",
    "ma_short":      "#1e88e5",
    "ma_medium":     "#fb8c00",
    "ma_long":       "#8e8e93",
    "macd":          "#1e88e5",
    "macd_signal":   "#fb8c00",
    "rsi":           "#6f42c1",
}
REGIME_COLORS = {
    "BULL": THEME["bull"],
    "SIDEWAYS": THEME["sideways"],
    "BEAR": THEME["bear"],
}
# 3-stop discrete colorscale for the signal heat-map. Hard cliffs at 1/3
# and 2/3 give us three flat bands (Sell / Hold / Buy) with no gradient.
SIGNAL_COLORSCALE = [
    [0.00, THEME["bear"]], [0.33, THEME["bear"]],
    [0.34, THEME["sideways"]], [0.66, THEME["sideways"]],
    [0.67, THEME["bull"]], [1.00, THEME["bull"]],
]
PLOTLY_TEMPLATE = "plotly_white"


def add_range_selector(fig: go.Figure, row: int | None = None,
                       col: int | None = None) -> go.Figure:
    """Attach TradingView-style range chips to a date x-axis.

    Buttons rendered: ``5D · 1M · 3M · 6M · YTD · 1Y · 3Y · MAX``.

    Plotly subtlety: ``fig.update_xaxes(row=, col=)`` only works on figures
    built with ``make_subplots`` (which set up an internal grid_ref). For
    **simple** ``go.Figure()`` charts (single axis), passing row/col raises
    an exception. So we pass row/col only when explicitly provided.
    """
    # Restored bordercolor/borderwidth (valid rangeselector properties that
    # give the chips their teal outline). The properties that caused the
    # Streamlit Cloud ValueError were xref/yref, not these.
    kwargs = dict(
        rangeselector=dict(
            buttons=[
                dict(count=5,  label="5D",  step="day",   stepmode="backward"),
                dict(count=1,  label="1M",  step="month", stepmode="backward"),
                dict(count=3,  label="3M",  step="month", stepmode="backward"),
                dict(count=6,  label="6M",  step="month", stepmode="backward"),
                dict(count=1,  label="YTD", step="year",  stepmode="todate"),
                dict(count=1,  label="1Y",  step="year",  stepmode="backward"),
                dict(count=3,  label="3Y",  step="year",  stepmode="backward"),
                dict(step="all", label="MAX"),
            ],
            bgcolor="rgba(15, 23, 42, 0.04)",
            activecolor=THEME["bull"],
            bordercolor=THEME["grid"],
            borderwidth=1,
            font=dict(size=11, color="#334155"),
            x=0.0, y=1.12, xanchor="left", yanchor="bottom",
        ),
        rangeslider=dict(visible=False),
    )
    if row is not None and col is not None:
        kwargs["row"] = row
        kwargs["col"] = col
    fig.update_xaxes(**kwargs)
    return fig


def style_fig(fig: go.Figure, height: int = 380, top_margin: int = 20) -> go.Figure:
    """Apply consistent margins / font / hover styling to any Plotly figure."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=40, r=20, t=top_margin, b=30),
        hoverlabel=dict(bgcolor="white", font_size=12),
        font=dict(family="Inter, system-ui, sans-serif", size=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0,
                    font=dict(size=11)),
    )
    fig.update_xaxes(showgrid=True, gridcolor=THEME["grid"], zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=THEME["grid"], zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# Cached pipeline
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_pipeline(universe: tuple[str, ...], start: str, end: str,
                 regime_method: str) -> dict:
    """End-to-end pipeline: load → features → regime → signals.

    Streamlit hashes the args to key the cache, so we pass tuples (hashable)
    not lists. The DataLoader's own on-disk CSV cache *also* avoids
    re-hitting yfinance on cold runs.
    """
    loader = DataLoader(universe=list(universe), start=start, end=end)
    data = loader.load_universe()
    benchmark = loader.load_benchmark()
    enriched = FeatureEngineer().compute_universe(data)
    det = RegimeDetector(method=regime_method)
    regimed = det.fit_transform_universe(enriched)
    signaled = SignalEngine().generate_universe(regimed)
    signaled = {t: add_entry_exit_levels(df) for t, df in signaled.items()}

    # Compute the *benchmark's* regime once here so the global status bar
    # can render it without paying that cost on every page render.
    bench_regime: str | None = None
    try:
        if benchmark is not None and not benchmark.empty:
            bench_enriched = FeatureEngineer().compute(benchmark)
            bench_labelled = RegimeDetector(method="hmm").fit_transform(
                bench_enriched
            )
            last = bench_labelled.dropna(subset=["Regime_Label"])
            if not last.empty:
                bench_regime = str(last["Regime_Label"].iloc[-1])
    except Exception as e:
        logger.warning("Benchmark regime computation failed: %s", e)

    return {
        "signaled": signaled,
        # `enriched` (features only — no regime/signal columns) is exposed
        # so the walk-forward engine can refit regimes per window without
        # contamination from the in-sample regime fit.
        "enriched": enriched,
        "benchmark": benchmark,
        "benchmark_regime": bench_regime,
        "quality": loader.get_quality_report(),
        "transition": det.get_transition_matrix(),
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def sidebar_controls() -> dict:
    st.sidebar.title("📈 AI Portfolio Advisor")
    st.sidebar.caption("NSE equities · NIFTY 50 benchmark")
    page = st.sidebar.radio(
        "Navigate",
        options=["Dashboard", "Stock Analyzer", "Portfolio Analyzer",
                 "Backtest Lab", "Recommendations"],
        index=0,
    )
    with st.sidebar.expander("⚙️ Pipeline settings", expanded=False):
        regime_method = st.selectbox(
            "Regime method", options=["hmm", "kmeans", "ma_crossover"], index=0,
            help="HMM is the production default (best bear-zone recall).",
        )
        start = st.date_input("Start date", value=date.fromisoformat(C.START_DATE))
        end = st.date_input("End date", value=date.fromisoformat(C.END_DATE))
        selected_universe = st.multiselect(
            "Universe", options=C.UNIVERSE, default=C.UNIVERSE,
            help="Trim the universe for faster runs.",
        )
    if not selected_universe:
        st.sidebar.error("Select at least one ticker.")
        st.stop()
    st.sidebar.divider()

    # ---- Live-data refresh control ----
    # The pipeline is heavy and cached; live quotes are light and short-TTL.
    # This button clears BOTH caches so the next render hits yfinance fresh.
    if st.sidebar.button("🔄 Force refresh data",
                          help=("Drops the pipeline cache AND the 60-second "
                                "live-quote cache, so the next page render "
                                "re-fetches the latest bars and LTPs.")):
        st.cache_data.clear()
        clear_live_cache()
        st.sidebar.success("Caches cleared. Reloading…")
        st.rerun()

    st.sidebar.caption(
        "ℹ️ Decision-support tool — not investment advice. Backtested "
        "performance does not guarantee future results. Live quotes are "
        "**~15–20 min delayed** (free Yahoo feed)."
    )
    cfg = {"page": page, "regime_method": regime_method,
           "start": start.isoformat(), "end": end.isoformat(),
           "universe": tuple(selected_universe)}
    # Stash so other pages (walk-forward) can read the regime method without
    # re-running the sidebar logic.
    st.session_state["last_cfg"] = cfg
    return cfg


# ---------------------------------------------------------------------------
# Small UI helpers
# ---------------------------------------------------------------------------
def metric_card(col, label: str, value: str, delta: str | None = None,
                help_text: str | None = None) -> None:
    col.metric(label=label, value=value, delta=delta, help=help_text)


# ---------------------------------------------------------------------------
# Indian rupee formatting — lakh / crore aware
# ---------------------------------------------------------------------------
def format_inr(amount, precision: int = 2) -> str:
    """Format a rupee amount using Indian lakh/crore notation when large.

    Rules:
        * ``< 1,00,000`` (1 lakh) → exact with thousand-separator commas
        * ``1,00,000 ≤ x < 1,00,00,000`` (1 cr) → ``"₹X.XX L"``
        * ``≥ 1,00,00,000``                    → ``"₹X.XX Cr"``

    Returns ``"—"`` for ``None`` / NaN / non-numeric inputs.

    Why this exists: Indian audiences (and recruiters scanning a CV
    project for the Indian market) read "₹2.48 L" *instantly*. They
    have to mentally translate "₹2,48,932" into a magnitude. The Indian
    lakh/crore convention is the universal tell that the project was
    built by someone who thinks in this market.
    """
    if amount is None:
        return "—"
    try:
        val = float(amount)
    except (TypeError, ValueError):
        return "—"
    if val != val:                                      # NaN
        return "—"
    sign = "-" if val < 0 else ""
    abs_val = abs(val)

    if abs_val < 100_000:
        # Small enough to show in full. For amounts < ₹100 keep decimals.
        if abs_val < 100:
            return f"{sign}₹{abs_val:.{precision}f}"
        return f"{sign}₹{abs_val:,.0f}"
    if abs_val < 10_000_000:
        return f"{sign}₹{abs_val / 100_000:.{precision}f} L"
    return f"{sign}₹{abs_val / 10_000_000:.{precision}f} Cr"


def format_inr_price(price, precision: int = 2) -> str:
    """Format a stock price — always show the exact value with decimals,
    never abbreviate. (₹3,025.50, not ₹3.03 K.)
    """
    if price is None:
        return "—"
    try:
        v = float(price)
    except (TypeError, ValueError):
        return "—"
    if v != v:
        return "—"
    sign = "-" if v < 0 else ""
    return f"{sign}₹{abs(v):,.{precision}f}"


def format_pct(value, precision: int = 2, signed: bool = True) -> str:
    """Format a fraction as a percentage with consistent sign handling."""
    if value is None:
        return "—"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if v != v:
        return "—"
    sign = "+" if (signed and v >= 0) else ""
    return f"{sign}{v * 100:.{precision}f}%"


def regime_badge(label: str) -> str:
    color = REGIME_COLORS.get(label, THEME["sideways"])
    return (
        f'<span style="background-color:{color};color:white;'
        f'padding:4px 12px;border-radius:14px;font-weight:600;'
        f'letter-spacing:0.5px;">{label}</span>'
    )


# ---------------------------------------------------------------------------
# Live-quote helpers (delayed-intraday, ~15-20 min latency)
# ---------------------------------------------------------------------------
# Two layers of caching:
#   1. src/live_quotes.py keeps an in-process dict cache with TTL = 60s.
#   2. Streamlit's @st.cache_data(ttl=60) wraps that so multiple components
#      on the same page don't each pay the per-ticker syscall cost.
@st.cache_data(ttl=60, show_spinner=False)
def cached_live_quotes(tickers: tuple[str, ...]) -> dict[str, dict]:
    """Cached batch live quotes. Returns plain dicts (Streamlit can't hash
    dataclasses straight from cache, so we go through ``__dict__``).
    """
    quotes = get_live_quotes(tickers)
    return {t: q.__dict__ for t, q in quotes.items()}


@st.cache_data(show_spinner=False)
def cached_lazy_load(
    tickers: tuple[str, ...], start: str, end: str, regime_method: str,
) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    """Cached lazy-load of extra tickers outside the curated universe.

    Cached so repeated portfolio analyses don't re-hit yfinance for the
    same off-universe symbols. Cache key is ``(tickers, start, end,
    regime_method)`` — re-running with the same args is instant.
    """
    return lazy_load_tickers(list(tickers), start=start, end=end,
                              regime_method=regime_method)


def extend_signaled(
    base_signaled: dict[str, pd.DataFrame],
    extra_tickers: list[str],
    cfg: dict,
) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    """Return a *new* signaled dict containing the base universe plus any
    successfully lazy-loaded extras. Never mutates ``base_signaled``.

    Returns ``(extended_signaled, succeeded, failed)``.
    """
    if not extra_tickers:
        return base_signaled, [], []
    extra, ok, bad = cached_lazy_load(
        tuple(sorted(set(extra_tickers))),
        start=cfg.get("start", C.START_DATE),
        end=cfg.get("end", C.END_DATE),
        regime_method=cfg.get("regime_method", "hmm"),
    )
    # Merge into a shallow copy so the cached pipe stays untouched.
    extended = {**base_signaled, **extra}
    return extended, ok, bad


def live_market_status_badge() -> str:
    """HTML pill — green if market is open, grey otherwise."""
    label, color = market_status_text()
    return (
        f'<span style="background-color:{color};color:white;'
        f'padding:4px 12px;border-radius:14px;font-weight:600;'
        f'letter-spacing:0.5px;font-size:0.85rem;">● {label}</span>'
    )


def _fmt_as_of(q: dict) -> str:
    """Short string describing the data's as-of date — e.g. 'Fri 28-May'."""
    d = q.get("as_of_date")
    if d is None:
        return ""
    try:
        return d.strftime("%a %d-%b")
    except Exception:
        return ""


def render_ticker_tape(tickers: tuple[str, ...], cols_per_row: int = 5) -> None:
    """Grid of small price cards for the given tickers.

    Three render modes per card:
      * **Live** (market open, fresh): white background, current LTP, change %.
      * **Previous close** (market closed, valid data): off-white background,
        "Close (DD-Mmm)" label, last full-day move shown as change %.
      * **Stale** (no data): grey, "—" placeholder.
    """
    quotes = cached_live_quotes(tickers)
    if not quotes:
        st.info("No price data available right now.")
        return

    rows = [list(tickers[i:i + cols_per_row])
            for i in range(0, len(tickers), cols_per_row)]
    for row in rows:
        cols = st.columns(len(row))
        for col, ticker in zip(cols, row):
            q = quotes.get(ticker)
            if q is None or q.get("stale"):
                col.markdown(
                    f"<div style='border:1px solid #e5e7eb;border-radius:8px;"
                    f"padding:8px 10px;background:#f3f4f6;'>"
                    f"<div style='font-size:0.78rem;color:#4b5563;'>{ticker}</div>"
                    f"<div style='font-size:1.05rem;font-weight:700;color:#9ca3af;'>—</div>"
                    f"<div style='font-size:0.75rem;color:#9ca3af;'>no data</div>"
                    "</div>", unsafe_allow_html=True,
                )
                continue
            change = float(q["change"])
            chg_pct = float(q["change_pct"])
            color = THEME["bull"] if change >= 0 else THEME["bear"]
            sign = "+" if change >= 0 else ""
            is_live = bool(q.get("is_live"))
            sub_label = (f"Live · {_fmt_as_of(q)}" if is_live
                          else f"Close · {_fmt_as_of(q)}")
            bg = "white" if is_live else "#f8fafc"
            # NOTE: every text element gets an EXPLICIT colour.
            # Streamlit's dark theme inherits a light text colour, which
            # made the price invisible against the white card background.
            col.markdown(
                f"<div style='border:1px solid #e5e7eb;border-radius:8px;"
                f"padding:8px 10px;background:{bg};'>"
                f"<div style='font-size:0.78rem;color:#4b5563;'>"
                f"{ticker}  <span style='float:right;color:#6b7280;font-size:0.7rem;'>"
                f"{sub_label}</span></div>"
                f"<div style='font-size:1.05rem;font-weight:700;color:#111827;'>"
                f"{format_inr_price(q['last_price'])}</div>"
                f"<div style='font-size:0.78rem;color:{color};font-weight:600;'>"
                f"{sign}{change:.2f}  ({sign}{chg_pct*100:.2f}%)</div>"
                "</div>",
                unsafe_allow_html=True,
            )


def render_top_status_bar(pipe: dict | None = None) -> None:
    """Bloomberg-inspired persistent strip shown on every page.

    Layout (compact, single row, dark navy background):
        ┌────────────────────────────────────────────────────────────────┐
        │ NIFTY 50 23,897 +0.78% │ SENSEX 78,452 +0.05% │ USD/INR ₹83.24 │
        │ INDIA VIX 14.2 │ Regime: BULL │ 10:46 IST │ ● MARKETS OPEN     │
        └────────────────────────────────────────────────────────────────┘

    Fetched live (delayed) on every page-render through ``cached_live_quotes``
    so navigating between pages is instant once warm. Tickers that fail to
    fetch are silently skipped — the bar never breaks the page.
    """
    headline_tickers = ("^NSEI", "^BSESN", "^INDIAVIX", "INR=X")
    quotes = cached_live_quotes(headline_tickers)

    def _cell(label: str, value: str, delta: str | None = None,
              delta_color: str | None = None) -> str:
        delta_html = ""
        if delta:
            color = delta_color or "#cbd5e1"
            delta_html = (f"<span style='color:{color};margin-left:6px;"
                          f"font-weight:600;font-size:0.82rem;'>{delta}</span>")
        # Tighter padding (was 2px 14px) so all 5 cells fit on one line
        # on standard widescreen viewports.
        return (
            f"<span style='display:inline-flex;align-items:baseline;"
            f"padding:2px 11px;border-right:1px solid #334155;"
            f"white-space:nowrap;'>"
            f"<span style='color:#94a3b8;font-size:0.70rem;"
            f"text-transform:uppercase;letter-spacing:0.06em;"
            f"font-weight:600;'>{label}</span>"
            f"<span style='font-weight:700;margin-left:7px;color:white;"
            f"font-size:0.90rem;'>{value}</span>"
            f"{delta_html}"
            f"</span>"
        )

    def _quote_cell(label: str, ticker: str, currency: str = "₹",
                    precision: int = 2) -> str:
        q = quotes.get(ticker)
        if q is None or q.get("stale"):
            return _cell(label, "—")
        lp = float(q["last_price"])
        pct = float(q["change_pct"]) * 100
        color = THEME["bull"] if pct >= 0 else THEME["bear"]
        sign = "+" if pct >= 0 else ""
        return _cell(label,
                     f"{currency}{lp:,.{precision}f}" if currency
                     else f"{lp:,.{precision}f}",
                     delta=f"{sign}{pct:.2f}%", delta_color=color)

    cells: list[str] = []
    cells.append(_quote_cell("NIFTY 50",   "^NSEI",     currency="", precision=0))
    cells.append(_quote_cell("SENSEX",     "^BSESN",    currency="", precision=0))
    cells.append(_quote_cell("USD/INR",    "INR=X",     currency="₹", precision=2))
    cells.append(_quote_cell("INDIA VIX",  "^INDIAVIX", currency="", precision=2))

    # Regime cell — read from the pre-computed benchmark regime if present.
    if pipe and pipe.get("benchmark_regime"):
        reg = pipe["benchmark_regime"]
        reg_color = REGIME_COLORS.get(reg, "#9ca3af")
        cells.append(_cell("REGIME", reg, delta_color=reg_color,
                            delta="●"))

    left_html = "".join(cells)

    # Right side: time + market-status pill.
    open_now = is_market_open()
    status_label, status_color = market_status_text()
    time_str = now_ist().strftime("%H:%M IST")
    right_html = (
        f"<span style='color:#cbd5e1;font-size:0.85rem;margin-right:12px;"
        f"font-weight:600;'>{time_str}</span>"
        f"<span style='background:{status_color};color:white;"
        f"padding:4px 12px;border-radius:12px;font-size:0.72rem;"
        f"font-weight:700;letter-spacing:0.06em;'>"
        f"● {status_label}</span>"
    )

    # Wrap behavior:
    #   * Left cluster (indices) shrinks/wraps as needed via flex-wrap.
    #   * Right cluster (time + status pill) uses margin-left: auto so it
    #     pushes to the END of its flex line even when wrapped — preventing
    #     the empty-space-on-the-right look when the bar splits onto two rows.
    st.markdown(
        f"""
        <div style='background:linear-gradient(90deg,#0f172a 0%,#1e293b 100%);
                    color:white;padding:12px 16px;border-radius:8px;
                    margin-top:4px;margin-bottom:1.4rem;
                    border:1px solid #334155;
                    box-shadow:0 2px 6px rgba(15,23,42,0.18);
                    font-variant-numeric: tabular-nums;
                    display:flex;align-items:center;flex-wrap:wrap;
                    row-gap:8px;column-gap:0;
                    min-height:46px;'>
            <div style='display:flex;flex-wrap:wrap;align-items:center;
                        line-height:1.4;flex:1 1 auto;'>
                {left_html}
            </div>
            <div style='display:flex;align-items:center;white-space:nowrap;
                        margin-left:auto;padding-left:14px;'>
                {right_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_header() -> None:
    """The standard band shown at the top of pages that benefit from a
    "this is current" cue. Adapts to market-open vs market-closed states.
    """
    from src.live_quotes import last_trading_day
    open_now = is_market_open()
    c1, c2 = st.columns([0.5, 0.5])
    with c1:
        st.markdown(live_market_status_badge(), unsafe_allow_html=True)
    with c2:
        if open_now:
            right = (f"Live (delayed ~15–20 min) · "
                     f"{now_ist():%d-%b-%Y %H:%M IST}")
        else:
            last_td = last_trading_day()
            right = (f"Showing last close · "
                     f"{last_td.strftime('%a %d-%b-%Y')}")
        st.markdown(
            f"<div style='text-align:right;color:#6b7280;font-size:0.8rem;'>"
            f"{right}</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Holdings upload widget (shared between Portfolio + Recommendations pages)
# ---------------------------------------------------------------------------
def holdings_uploader(key_prefix: str,
                      default_text: str = "") -> dict[str, float]:
    """Return parsed ``{ticker: amount}`` from upload OR text input.

    Provides three input methods in tabs so the user picks whichever is
    convenient: file upload (CSV/XLSX/PDF), paste-as-text, or template
    download.
    """
    tab_upload, tab_text, tab_template = st.tabs([
        "📁 Upload (CSV / Excel / PDF)",
        "⌨️ Type / Paste",
        "📥 Download template",
    ])

    holdings: dict[str, float] = {}

    with tab_upload:
        st.caption(
            "Upload a broker export. Auto-detects whichever schema you have: "
            "**Ticker + Amount** or **Ticker + Quantity + Avg Price**. "
            "Synonyms recognised (Symbol/Scrip, Value/Investment, Qty, etc.)."
        )
        up = st.file_uploader(
            "Choose file", type=["csv", "xlsx", "xls", "pdf"],
            key=f"{key_prefix}_upload",
            help="CSV and Excel: first sheet only. PDF: tables auto-extracted.",
        )
        if up is not None:
            try:
                with st.spinner(f"Parsing {up.name}…"):
                    parsed = parse_holdings_file(up, filename=up.name)
                st.success(f"Parsed {len(parsed)} holdings from **{up.name}**.")
                preview = (pd.DataFrame(
                    [{"Ticker": k, "Amount (₹)": v} for k, v in parsed.items()])
                    .sort_values("Amount (₹)", ascending=False))
                st.dataframe(
                    preview.style.format({"Amount (₹)": "{:,.0f}"}),
                    use_container_width=True, hide_index=True, height=240,
                )
                if st.button("✅ Use these holdings", key=f"{key_prefix}_use",
                             type="primary"):
                    holdings = parsed
                    st.session_state[f"{key_prefix}_holdings"] = parsed
            except Exception as e:
                st.error(f"Couldn't parse the file: {e}")
                st.info(
                    "Tip: the file needs a column for the ticker (Symbol / "
                    "Ticker / Scrip…) and either an Amount column or a "
                    "Quantity + Avg-Price pair. See the **Download template** "
                    "tab for a working example."
                )

    with tab_text:
        raw = st.text_area(
            "One `TICKER, AMOUNT` per line",
            value=default_text, height=200, key=f"{key_prefix}_textarea",
        )
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                t, a = [x.strip() for x in line.split(",", 1)]
                holdings[t] = float(a.replace(",", "").replace("₹", "").replace("Rs", ""))
            except Exception:
                pass

    with tab_template:
        st.caption("Download a starter template, edit the rows, then upload it back.")
        tmpl_compact = pd.DataFrame({
            "Ticker": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"],
            "Amount": [30_000, 25_000, 25_000, 20_000],
        })
        tmpl_detailed = pd.DataFrame({
            "Ticker": ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS"],
            "Quantity": [10, 5, 15, 12],
            "Avg Price": [3_000.0, 5_000.0, 1_650.0, 1_650.0],
        })
        c1, c2 = st.columns(2)
        c1.download_button(
            "⬇️ Compact (Ticker + Amount)",
            data=tmpl_compact.to_csv(index=False),
            file_name="holdings_template_compact.csv", mime="text/csv",
            use_container_width=True, key=f"{key_prefix}_tmpl_compact",
        )
        c2.download_button(
            "⬇️ Detailed (Ticker + Qty + Avg Price)",
            data=tmpl_detailed.to_csv(index=False),
            file_name="holdings_template_detailed.csv", mime="text/csv",
            use_container_width=True, key=f"{key_prefix}_tmpl_detailed",
        )

    # Session state lets the "Use these holdings" button persist across reruns.
    if not holdings and f"{key_prefix}_holdings" in st.session_state:
        holdings = st.session_state[f"{key_prefix}_holdings"]
    return holdings


# ---------------------------------------------------------------------------
# Page 1 — Dashboard
# ---------------------------------------------------------------------------
def page_dashboard(pipe: dict) -> None:
    st.title("Dashboard")

    # Live market-status header (open/closed + IST timestamp + delay note).
    render_live_header()

    bench = pipe["benchmark"]
    signaled = pipe["signaled"]

    # --- Re-run regime detector on the benchmark for a "market view" ---
    bench_enriched = FeatureEngineer().compute(bench)
    bench_labelled = RegimeDetector(method="hmm").fit_transform(bench_enriched)
    latest = bench_labelled.dropna(subset=["Close", "Regime_Label"]).iloc[-1]
    latest_label = str(latest["Regime_Label"])

    st.markdown(f"### Market regime: {regime_badge(latest_label)}",
                unsafe_allow_html=True)

    # --- NIFTY 50 quote (label adapts to market state) + HMM posteriors ---
    nifty_quote = cached_live_quotes((C.BENCHMARK,)).get(C.BENCHMARK)
    c1, c2, c3, c4 = st.columns(4)
    if nifty_quote and not nifty_quote.get("stale"):
        delta = (f"{'+' if nifty_quote['change'] >= 0 else ''}"
                 f"{nifty_quote['change']:.2f} "
                 f"({nifty_quote['change_pct']*100:+.2f}%)")
        # Label changes between live and previous-close states.
        if nifty_quote.get("is_live"):
            label = "NIFTY 50 LTP (delayed)"
        else:
            label = f"NIFTY 50 close · {_fmt_as_of(nifty_quote)}"
        metric_card(c1, label,
                    format_inr_price(nifty_quote['last_price'], precision=0),
                    delta=delta)
    else:
        metric_card(c1, "NIFTY 50 (last close)",
                    format_inr_price(latest['Close'], precision=0),
                    help_text="Live quote unavailable; showing last EOD close.")
    if "Regime_Prob_Bull" in bench_labelled.columns:
        metric_card(c2, "P(Bull) [HMM]", f"{float(latest['Regime_Prob_Bull'])*100:.1f}%")
        metric_card(c3, "P(Bear) [HMM]", f"{float(latest['Regime_Prob_Bear'])*100:.1f}%")
    metric_card(c4, "Universe tickers", f"{len(signaled)}")

    # --- Ticker tape (label adapts to market state) ---
    if is_market_open():
        st.subheader("Universe — live (delayed) quotes")
    else:
        st.subheader("Universe — last close")
    universe_tickers = tuple(sorted(signaled.keys()))[:15]   # cap to keep it readable
    render_ticker_tape(universe_tickers, cols_per_row=5)

    # --- Top BUY signals ---
    st.subheader("Top BUY signals (latest bar)")
    rows = []
    for t, df in signaled.items():
        last = df.dropna(subset=["Signal"]).tail(1)
        if last.empty:
            continue
        r = last.iloc[0]
        if r.get("Signal", 0) == 1:
            rows.append({
                "Ticker": t,
                "Close": float(r["Close"]),
                "Confidence": float(r["Confidence"]),
                "Strength": str(r["Signal_Strength"]),
                "Regime": str(r["Regime_Label"]),
                "Stop": float(r.get("Stop_Loss", np.nan)),
                "Target 1": float(r.get("Target_1", np.nan)),
                "Target 2": float(r.get("Target_2", np.nan)),
                "R/R": float(r.get("Risk_Reward", 0.0)),
                "As of": last.index[0].date(),
            })
    if rows:
        top = pd.DataFrame(rows).sort_values("Confidence", ascending=False).head(10)
        st.dataframe(
            top.style.format({
                "Close": "{:,.1f}", "Confidence": "{:.2f}",
                "Stop": "{:,.1f}", "Target 1": "{:,.1f}",
                "Target 2": "{:,.1f}", "R/R": "{:.1f}",
            }),
            use_container_width=True, hide_index=True,
        )
    else:
        st.info("No active BUY signals on the latest bar.")

    # --- Fixed signal heat-map ---
    st.subheader("Signal heat-map — last 30 trading days")
    st.caption(
        "Each row is a ticker, each cell is a day. "
        f"🟢 = Buy, ⚪ = Hold, 🔴 = Sell. Hover for details."
    )
    fig = build_signal_heatmap(signaled, lookback=30)
    if fig is None:
        st.info("Not enough data to render the heat-map.")
    else:
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Data-quality report", expanded=False):
        st.dataframe(pipe["quality"], use_container_width=True, hide_index=True)


def build_signal_heatmap(signaled: dict[str, pd.DataFrame],
                         lookback: int = 30) -> go.Figure | None:
    """Discrete 3-color signal heat-map with real date ticks + rich hover.

    Issues with the previous version:
      * X-axis just said "Last 30 trading bars" — no actual dates.
      * Continuous colorscale interpolated between -1, 0, +1 (gradient
        rather than three flat bands).
      * Hover only showed the raw value.

    Fixes:
      * Discrete colorscale with hard cliffs (see ``SIGNAL_COLORSCALE``).
      * Real date strings on the x-axis (formatted ``DD-Mon``).
      * ``customdata`` carries ticker + date + confidence + label so the
        hover tooltip is human-readable.
    """
    z_rows, conf_rows, label_rows, tickers, date_strs = [], [], [], [], []
    # Align to the union of recent dates so columns line up across tickers.
    # Defensive: normalise to tz-naive so mixed tz indices (from cache vs
    # fresh fetch) don't break the sort with a TypeError.
    def _strip_tz(d):
        ts = pd.Timestamp(d)
        return ts.tz_localize(None) if ts.tz is not None else ts

    all_dates = sorted({
        _strip_tz(d)
        for df in signaled.values()
        for d in df.dropna(subset=["Signal"]).tail(lookback).index
    })
    if not all_dates:
        return None
    recent = pd.DatetimeIndex(all_dates[-lookback:])
    date_strs = [d.strftime("%d-%b") for d in recent]

    for t, df in signaled.items():
        sub = df.reindex(recent)
        sig = sub["Signal"].astype(float)
        conf = sub["Confidence"].astype(float)
        if sig.dropna().empty:
            continue
        tickers.append(t)
        z_rows.append(sig.values)
        conf_rows.append(conf.values)
        labels = sub["Signal_Strength"].fillna("—").astype(str).values
        label_rows.append(labels)

    if not z_rows:
        return None

    z = np.array(z_rows, dtype=float)
    # customdata: stack confidence + label per cell for hovertemplate.
    cdata = np.stack([np.array(conf_rows, dtype=float),
                      np.array(label_rows, dtype=object)], axis=-1)

    fig = go.Figure(go.Heatmap(
        z=z, x=date_strs, y=tickers,
        zmin=-1, zmax=1,
        colorscale=SIGNAL_COLORSCALE,
        colorbar=dict(
            title="Signal", tickvals=[-1, 0, 1],
            ticktext=["Sell", "Hold", "Buy"],
            len=0.6, thickness=14,
        ),
        xgap=1, ygap=1,        # thin white grid for cell separation
        customdata=cdata,
        hovertemplate=(
            "<b>%{y}</b> · %{x}"
            "<br>Signal: <b>%{customdata[1]}</b>"
            "<br>Confidence: %{customdata[0]:.2f}"
            "<extra></extra>"
        ),
    ))
    fig.update_layout(
        height=max(360, 22 * len(tickers) + 80),
        margin=dict(l=90, r=10, t=10, b=40),
        template=PLOTLY_TEMPLATE,
        font=dict(family="Inter, system-ui, sans-serif", size=11),
        xaxis=dict(side="bottom", tickangle=-45),
        yaxis=dict(autorange="reversed"),     # alphabetic top → bottom feels natural
    )
    return fig


# ---------------------------------------------------------------------------
# Page 2 — Stock Analyzer
# ---------------------------------------------------------------------------
def page_stock_analyzer(pipe: dict) -> None:
    st.title("Stock Analyzer")
    signaled = pipe["signaled"]
    if not signaled:
        st.warning("No tickers loaded.")
        return

    render_live_header()

    # ---- Ticker picker — three modes ----
    # 1. Universe   → fast dropdown of the 25 curated stocks
    # 2. Search NSE → searchable selectbox over ~200 named NSE stocks
    #                 (type "avenue" → DMART.NS — Avenue Supermarts shows up)
    # 3. Custom     → free-text input for symbols not in our catalogue
    mode = st.radio(
        "Source",
        ["Universe", "🔎 Search NSE (by name or ticker)", "Type any symbol"],
        horizontal=True, key="stock_picker_mode",
        help=("Universe = the 25 curated stocks. "
              "Search NSE = browse ~200 named stocks by company name. "
              "Type any symbol = enter any NSE ticker not in our catalogue."),
    )

    ticker: str | None = None

    if mode == "Universe":
        ticker = st.selectbox("Ticker", options=sorted(signaled.keys()))

    elif mode.startswith("🔎"):
        # Pre-formatted "TICKER — Company Name" options, alphabetised by name.
        # Streamlit's selectbox filters as you type, so "avenue" → DMART.NS.
        search_options = [
            f"{t.replace('.NS', '').replace('^', '')}  —  {n}"
            for t, n in sorted(NSE_STOCK_NAMES.items(), key=lambda kv: kv[1])
        ]
        choice = st.selectbox(
            "Start typing a company name or ticker",
            options=["— start typing —"] + search_options,
            help=("Try: 'avenue' → Avenue Supermarts · 'tata' → Tata stocks · "
                  "'bank' → all listed banks · 'paytm' → One 97 Communications"),
            key="stock_picker_search",
        )
        if choice == "— start typing —":
            st.info(
                "💡 Type to filter — e.g. **avenue** → Avenue Supermarts, "
                "**tata** → Tata group, **paytm** → Paytm. "
                "If your stock isn't here, switch to **Type any symbol**."
            )
            return
        # Extract the ticker from "TICKER  —  Company Name".
        ticker_part = choice.split("—")[0].strip()
        ticker = normalise_ticker(ticker_part)

    else:                                # "Type any symbol"
        raw = st.text_input(
            "Symbol (no need to add .NS)", value="",
            placeholder="e.g. BAJAJ-AUTO, MOTHERSON, IRCTC",
            key="stock_picker_custom",
        )
        ticker = normalise_ticker(raw)
        if not ticker:
            st.info("Type a symbol above to begin.")
            return

    # Lazy-load the ticker if it's not already in the cached pipeline.
    if ticker not in signaled:
        cfg = st.session_state.get("last_cfg", {})
        with st.spinner(f"📡 Fetching {ticker} from yfinance…"):
            signaled, ok, bad = extend_signaled(signaled, [ticker], cfg)
        if ticker in bad or ticker not in signaled:
            st.error(
                f"Couldn't load **{ticker}** — yfinance returned nothing, "
                "or it failed the QA gate (need ≥200 trading days, <5% "
                "missing days). Check the spelling, or try a different symbol."
            )
            return
        st.success(f"Loaded {ticker} ({len(signaled[ticker])} bars).")

    df = signaled[ticker].dropna(subset=["Close"]).copy()
    if df.empty:
        st.warning(f"No data for {ticker}.")
        return

    last = df.iloc[-1]
    # ---- Price card row ----
    # Label adapts to whether the ticker is live (market open) or showing
    # the previous-session close (weekend / after-hours).
    live = cached_live_quotes((ticker,)).get(ticker)
    c1, c2, c3, c4, c5 = st.columns(5)
    if live and not live.get("stale"):
        delta = (f"{'+' if live['change'] >= 0 else ''}"
                 f"{live['change']:.2f}  "
                 f"({format_pct(live['change_pct'])})")
        if live.get("is_live"):
            label = "LTP (delayed)"
        else:
            label = f"Close · {_fmt_as_of(live)}"
        metric_card(c1, label, format_inr_price(live['last_price']), delta=delta)
    else:
        metric_card(c1, "Last close (EOD)", format_inr_price(last['Close']),
                    help_text="Price quote unavailable; showing latest EOD close.")
    metric_card(c2, "Signal", str(last.get("Signal_Strength", "-")))
    metric_card(c3, "Confidence", f"{float(last.get('Confidence', 0))*100:.0f}%")
    metric_card(c4, "Regime", str(last.get("Regime_Label", "-")))
    metric_card(c5, "RSI 14", f"{float(last.get('RSI_14', np.nan)):.1f}")

    # Day's range strip — only when we have a fresh quote with day high/low set.
    if live and not live.get("stale") and live.get("day_high") and live.get("day_low"):
        prefix = "📡 Today's range" if live.get("is_live") else "📅 Session range"
        st.caption(
            f"{prefix}: {format_inr_price(live['day_low'])} — "
            f"{format_inr_price(live['day_high'])}  ·  "
            f"Prev close: {format_inr_price(live['previous_close'])}  ·  "
            f"Vol: {float(live.get('volume') or 0):,.0f}"
        )

    # ---- Range chip menu (Streamlit-side, pill-shaped) ----
    # Why this and not Plotly's rangeselector: Plotly's built-in chips have
    # a documented bug with `shared_xaxes=True` subplots on the version
    # Streamlit Cloud runs — clicks on 1M/3M wouldn't propagate to the
    # visible bottom axis. We bypass that by filtering the dataframe in
    # Streamlit before building the chart, guaranteeing all 4 rows zoom
    # together.
    _RANGES_DAYS: dict[str, int | None] = {
        "5D": 5, "1M": 22, "3M": 66, "6M": 132,
        "YTD": None, "1Y": 252, "3Y": 252 * 3, "MAX": None,
    }
    _picker = getattr(st, "segmented_control", None)
    if _picker is not None:
        range_choice = _picker(
            "Range", options=list(_RANGES_DAYS.keys()),
            default="1Y", key=f"stock_range_{ticker}",
            label_visibility="collapsed",
        )
    else:
        # Fallback for older Streamlit versions.
        range_choice = st.radio(
            "Range", options=list(_RANGES_DAYS.keys()),
            index=5, horizontal=True,
            key=f"stock_range_{ticker}",
            label_visibility="collapsed",
        )
    if not range_choice:
        range_choice = "1Y"

    if range_choice == "MAX" or df.empty:
        df_view = df
    elif range_choice == "YTD":
        ytd_start = pd.Timestamp(df.index[-1].year, 1, 1)
        df_view = df.loc[df.index >= ytd_start]
        if df_view.empty:
            df_view = df
    else:
        n = _RANGES_DAYS[range_choice]
        df_view = df.iloc[-n:] if (n is not None and len(df) > n) else df
    df = df_view

    # ---- Combined 4-row chart: price/volume on top, RSI, MACD ----
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.025,
        row_heights=[0.5, 0.15, 0.175, 0.175],
        subplot_titles=("Price + MAs", "Volume", "RSI 14", "MACD"),
    )

    # Row 1 — candles
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="OHLC",
        increasing_line_color=THEME["bull"],
        decreasing_line_color=THEME["bear"],
        showlegend=False,
    ), row=1, col=1)
    for col, colour in (("MA_20", THEME["ma_short"]),
                        ("MA_50", THEME["ma_medium"]),
                        ("MA_200", THEME["ma_long"])):
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                                     line=dict(color=colour, width=1.4)),
                          row=1, col=1)

    # Signal markers — off-bar for legibility.
    buys = df[df["Signal"] == 1]
    sells = df[df["Signal"] == -1]
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=buys.index, y=buys["Low"] * 0.985, mode="markers",
            marker=dict(symbol="triangle-up", color=THEME["bull"], size=11,
                        line=dict(width=1, color="white")),
            name="BUY",
            customdata=np.stack([buys["Confidence"].values,
                                 buys["Signal_Strength"].values], axis=-1),
            hovertemplate=("BUY · %{x|%d-%b-%Y}"
                           "<br>%{customdata[1]}"
                           "<br>Conf: %{customdata[0]:.2f}<extra></extra>"),
        ), row=1, col=1)
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=sells.index, y=sells["High"] * 1.015, mode="markers",
            marker=dict(symbol="triangle-down", color=THEME["bear"], size=11,
                        line=dict(width=1, color="white")),
            name="SELL",
            customdata=np.stack([sells["Confidence"].values,
                                 sells["Signal_Strength"].values], axis=-1),
            hovertemplate=("SELL · %{x|%d-%b-%Y}"
                           "<br>%{customdata[1]}"
                           "<br>Conf: %{customdata[0]:.2f}<extra></extra>"),
        ), row=1, col=1)

    # Row 2 — Volume (integrated; previously a separate sub-chart)
    if "Volume" in df.columns:
        # Color volume bars by daily direction.
        up = df["Close"] >= df["Open"]
        vol_colors = np.where(up, THEME["bull"], THEME["bear"])
        fig.add_trace(go.Bar(
            x=df.index, y=df["Volume"], marker_color=vol_colors,
            marker_line_width=0, opacity=0.6, name="Volume", showlegend=False,
            hovertemplate="%{x|%d-%b-%Y}<br>Vol: %{y:,.0f}<extra></extra>",
        ), row=2, col=1)

    # Row 3 — RSI with shaded 30/70 zones.
    if "RSI_14" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI_14"], name="RSI 14",
                                 line=dict(color=THEME["rsi"], width=1.5),
                                 showlegend=False), row=3, col=1)
        fig.add_hrect(y0=70, y1=100, fillcolor=THEME["bear"], opacity=0.08,
                      line_width=0, row=3, col=1)
        fig.add_hrect(y0=0, y1=30, fillcolor=THEME["bull"], opacity=0.08,
                      line_width=0, row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=THEME["bear"], row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=THEME["bull"], row=3, col=1)

    # Row 4 — MACD + signal line + histogram.
    if "MACD" in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD",
                                 line=dict(color=THEME["macd"], width=1.4),
                                 showlegend=False), row=4, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal",
                                 line=dict(color=THEME["macd_signal"], width=1.4,
                                           dash="dot"), showlegend=False), row=4, col=1)
        hist = df["MACD_Hist"].fillna(0)
        colors = np.where(hist >= 0, THEME["bull"], THEME["bear"])
        fig.add_trace(go.Bar(x=df.index, y=hist, name="Histogram",
                             marker_color=colors, marker_line_width=0,
                             opacity=0.55, showlegend=False), row=4, col=1)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=900, margin=dict(l=40, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False, hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.0),
        font=dict(family="Inter, system-ui, sans-serif", size=12),
    )
    for r in (1, 2, 3, 4):
        fig.update_xaxes(showgrid=True, gridcolor=THEME["grid"], row=r, col=1)
        fig.update_yaxes(showgrid=True, gridcolor=THEME["grid"], row=r, col=1)
    st.plotly_chart(fig, use_container_width=True)

    # Trade plan card on active signals.
    sig = int(last.get("Signal", 0))
    if sig != 0:
        verb = "BUY" if sig > 0 else "SELL"
        st.subheader(f"Trade plan (latest {verb})")
        cols = st.columns(4)
        metric_card(cols[0], "Entry", format_inr_price(last['Entry_Price']))
        metric_card(cols[1], "Stop-loss", format_inr_price(last['Stop_Loss']))
        metric_card(cols[2], "Target 1", format_inr_price(last['Target_1']))
        metric_card(cols[3],
                    f"Target 2 (R/R {float(last['Risk_Reward']):.1f})",
                    format_inr_price(last['Target_2']))


# ---------------------------------------------------------------------------
# Page 3 — Portfolio Analyzer
# ---------------------------------------------------------------------------
def page_portfolio_analyzer(pipe: dict) -> None:
    st.title("Portfolio Analyzer")
    signaled = pipe["signaled"]

    holdings = holdings_uploader(
        key_prefix="port",
        default_text="RELIANCE.NS, 30000\nTCS.NS, 25000\nHDFCBANK.NS, 25000\nINFY.NS, 20000",
    )

    if not holdings:
        st.info("👆 Upload a file, paste your holdings, or use the template to begin.")
        return

    missing = [t for t in holdings if t not in signaled]
    if missing:
        # Two distinct cases:
        #   (a) Ticker IS in C.UNIVERSE but failed to load this run (yfinance
        #       hiccup / QA gate rejection). Skip with warning.
        #   (b) Ticker is OUTSIDE C.UNIVERSE. Lazy-fetch it via dynamic_universe.
        in_universe_failed = [t for t in missing if t in C.UNIVERSE]
        outside_universe = [t for t in missing if t not in C.UNIVERSE]

        if in_universe_failed:
            st.warning(
                f"⚠️ {len(in_universe_failed)} ticker(s) are in the universe but "
                f"didn't load this run: **{in_universe_failed}**. Check the "
                "**Dashboard → Data-quality report** for the exact reason. "
                "Skipping them."
            )

        if outside_universe:
            cfg = st.session_state.get("last_cfg", {})
            with st.spinner(
                f"📡 Fetching {len(outside_universe)} ticker(s) outside the "
                f"default universe — {', '.join(outside_universe[:4])}"
                f"{'…' if len(outside_universe) > 4 else ''}"
            ):
                signaled, fetched, failed_fetch = extend_signaled(
                    signaled, outside_universe, cfg,
                )
            if fetched:
                st.success(
                    f"✅ Loaded **{len(fetched)}** additional ticker(s): "
                    f"{', '.join(fetched)}. Continuing with full analysis."
                )
            if failed_fetch:
                st.warning(
                    f"⚠️ Couldn't load **{failed_fetch}** — either yfinance has "
                    "no data or the QA gate rejected them (too few bars, recent "
                    "IPO, delisted, etc.). Skipping these holdings."
                )

        # Drop anything we still couldn't resolve.
        holdings = {t: a for t, a in holdings.items() if t in signaled}
        if not holdings:
            st.error("None of your holdings loaded. Cannot analyze.")
            return

    if not st.button("🔍 Analyze portfolio", type="primary"):
        return

    pa = PortfolioAnalyzer(holdings)
    with st.spinner("Computing risk metrics…"):
        pa.analyze(signaled)
    m = pa.portfolio_metrics
    d = pa.diversification
    v = pa.var

    # ------------------ Live portfolio value (delayed-intraday) ------------------
    # We interpret the uploaded "Amount" as the position's value at the most
    # recent EOD close. Implied shares = amount / last_close → live value =
    # implied_shares × LTP. This gives an honest "since end-of-day" delta
    # without needing the user's actual cost basis.
    render_live_header()
    live_value, live_delta_inr, live_delta_pct = _compute_live_portfolio_value(
        holdings, signaled,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    metric_card(c1, "Cost basis (uploaded)", format_inr(m['total_value']))
    if live_value is not None:
        delta_str = (f"{format_inr(live_delta_inr)}  "
                     f"({format_pct(live_delta_pct)})")
        if is_market_open():
            label = "Live value (intraday)"
            tip = "Marked to delayed LTP. Updates ~every 60s with cache refresh."
        else:
            label = "Latest close value"
            tip = ("Marked to most recent session's close. Live updates resume "
                   "when markets reopen.")
        metric_card(c2, label, format_inr(live_value),
                    delta=delta_str, help_text=tip)
    else:
        metric_card(c2, "Live value", "—",
                    help_text="Live quotes unavailable right now.")
    metric_card(c3, "Ann. return", format_pct(m['ann_return']))
    metric_card(c4, "Ann. volatility", format_pct(m['ann_vol'], signed=False),
                help_text="Markowitz σ_p = √(wᵀ Σ w)")
    metric_card(c5, "Sharpe", f"{m['sharpe']:.2f}")

    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Diversification", f"{d['score']:.0f} / 100",
                delta=d["label"])
    metric_card(c2, "Mean correlation", f"{d['mean_pairwise_correlation']:.2f}")
    metric_card(c3, "VaR 95% (1d)", format_inr(abs(v['var_95_inr'])),
                help_text=f"{v['var_95_pct']*100:.2f}% historical")
    metric_card(c4, "CVaR 95% (Expected Shortfall)",
                format_inr(abs(v['cvar_95_inr'])))

    # ------------------ Sector treemap + correlation heatmap ------------------
    left, right = st.columns([0.55, 0.45])
    with left:
        st.subheader("Sector exposure")
        tm = sector_treemap(pa.sector_exposure, pa.stock_metrics)
        st.plotly_chart(tm, use_container_width=True)
    with right:
        st.subheader("Correlation heat-map")
        st.plotly_chart(correlation_heatmap(pa.corr), use_container_width=True)

    # ------------------ Per-stock tables in expanders ------------------
    with st.expander("📊 Per-stock breakdown", expanded=True):
        sm = pa.stock_metrics.copy()
        st.dataframe(
            sm.style.format({
                "weight": "{:.1%}",
                "value": format_inr,
                "ann_return": "{:.1%}",
                "ann_vol": "{:.1%}",
                "sharpe": "{:.2f}",
                "max_drawdown": "{:.1%}",
            }),
            use_container_width=True, hide_index=True,
        )

    with st.expander("⚖️ Risk contributions", expanded=False):
        rc = pa.risk_contrib.copy()
        st.dataframe(
            rc.style.format({
                "weight": "{:.1%}", "risk_contribution": "{:.3f}",
                "risk_contribution_pct": "{:.1%}", "risk_vs_weight": "{:.2f}×",
            }),
            use_container_width=True, hide_index=True,
        )
        st.caption(
            "Risk-vs-weight > 1.5 → 'risk hog' — a position contributing more "
            "to portfolio volatility than its cash weight."
        )

    # ------------------ Warnings ------------------
    st.subheader("Warnings")
    if not pa.warnings:
        st.success("✓ No structural warnings — the portfolio looks healthy.")
    else:
        for w in pa.warnings:
            badge = "🟥" if w["severity"] == "HIGH" else "🟧"
            st.markdown(f"{badge} **{w['type']}** — {w['message']}")


def _compute_live_portfolio_value(
    holdings: dict[str, float],
    signaled: dict[str, pd.DataFrame],
) -> tuple[float | None, float, float]:
    """Mark each holding to live LTP and return (total, ₹ delta, % delta).

    Implied shares per holding = ``amount / last_close``. Live value of the
    holding = implied_shares × LTP. The aggregate live value is summed across
    all holdings; the delta is computed against the uploaded cost basis.

    Returns ``(None, 0.0, 0.0)`` if every live quote is stale — callers
    render an unobtrusive '—' instead.
    """
    if not holdings:
        return None, 0.0, 0.0
    quotes = cached_live_quotes(tuple(holdings.keys()))
    cost_basis = float(sum(holdings.values()))
    live_total = 0.0
    fresh_any = False
    for ticker, amount in holdings.items():
        q = quotes.get(ticker)
        df = signaled.get(ticker)
        last_close = (float(df["Close"].dropna().iloc[-1])
                      if df is not None and not df.empty else None)
        if q is None or q.get("stale") or last_close is None or last_close <= 0:
            live_total += amount                    # no fresh data → no change
            continue
        implied_shares = amount / last_close
        live_total += implied_shares * float(q["last_price"])
        fresh_any = True
    if not fresh_any:
        return None, 0.0, 0.0
    delta_inr = live_total - cost_basis
    delta_pct = delta_inr / cost_basis if cost_basis > 0 else 0.0
    return live_total, delta_inr, delta_pct


def sector_treemap(sector_df: pd.DataFrame,
                   stock_df: pd.DataFrame) -> go.Figure:
    """Hierarchical sector → ticker treemap (replaces the pie chart).

    Why a treemap: pie charts compare proportions; treemaps let you see
    the *structure* (sector dominance + which tickers drive each sector)
    in one glance. Far more useful for a portfolio audit.
    """
    rows = stock_df.merge(sector_df[["sector"]].assign(_=1),
                          on="sector", how="left")
    fig = px.treemap(
        rows, path=[px.Constant("Portfolio"), "sector", "ticker"],
        values="weight",
        color="ann_return",
        color_continuous_scale=[
            (0.0, THEME["bear"]), (0.5, THEME["sideways"]), (1.0, THEME["bull"]),
        ],
        color_continuous_midpoint=0.0,
        hover_data={"weight": ":.1%", "ann_return": ":.1%", "ann_vol": ":.1%"},
    )
    fig.update_traces(textinfo="label+percent parent",
                      hovertemplate="<b>%{label}</b><br>Weight: %{value:.1%}"
                                    "<br>Ann. return: %{color:.1%}<extra></extra>")
    fig.update_layout(template=PLOTLY_TEMPLATE, height=380,
                      margin=dict(l=0, r=0, t=0, b=0),
                      coloraxis_colorbar=dict(title="Ann. return",
                                              tickformat=".0%", len=0.7))
    return fig


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    """Symmetric correlation matrix with annotations on small portfolios.

    Annotation labels are dropped for >10 tickers (gets unreadable).
    """
    show_labels = len(corr) <= 10
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index,
        zmin=-1, zmax=1, colorscale="RdBu_r",
        colorbar=dict(title="ρ", len=0.7, thickness=12),
        text=np.round(corr.values, 2) if show_labels else None,
        texttemplate="%{text}" if show_labels else None,
        textfont={"size": 10},
        hovertemplate="<b>%{y} ↔ %{x}</b><br>ρ = %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=380,
                      margin=dict(l=10, r=10, t=10, b=10),
                      yaxis=dict(autorange="reversed"))
    return fig


# ---------------------------------------------------------------------------
# Page 4 — Backtest Lab
# ---------------------------------------------------------------------------
def page_backtest_lab(pipe: dict) -> None:
    """Backtest Lab — restructured into 3 tabs:

      1. Performance — equity, drawdown, monthly heat-map, trade P&L (existing).
      2. Statistical Significance — bootstrap CI, PSR, DSR, alpha t-stat (new).
      3. Walk-Forward OOS — refit per window, OOS equity, IS vs OOS comparison (new).

    The backtest itself runs ONCE per parameter set; the result is cached in
    ``st.session_state`` so navigating between the three tabs is instant.
    """
    st.title("Backtest Lab")
    signaled = pipe["signaled"]
    benchmark = pipe["benchmark"]

    # ---- Configuration row ----
    c1, c2, c3 = st.columns(3)
    capital = c1.number_input("Initial capital (₹)", value=int(C.INITIAL_CAPITAL),
                              step=10_000, min_value=10_000)
    txn = c2.number_input("Transaction cost (per side)", value=float(C.TRANSACTION_COST),
                          step=0.0001, format="%.4f")
    max_pos = c3.number_input("Max open positions", value=int(C.MAX_OPEN_POSITIONS),
                              step=1, min_value=1, max_value=20)

    if st.button("▶️ Run backtest", type="primary"):
        bt = Backtester(initial_capital=capital, transaction_cost=txn,
                        max_positions=int(max_pos))
        with st.spinner("Running event-driven simulation…"):
            bt.run(signaled, benchmark_df=benchmark)
        # Cache the result so tab-switches and walk-forward inherit it.
        st.session_state["bt_result"] = bt
        st.session_state["bt_params"] = {"capital": capital, "txn": txn,
                                          "max_pos": max_pos}
        # Clear any stale walk-forward / significance cache on re-run.
        st.session_state.pop("wf_result", None)
        st.session_state.pop("sig_result", None)

    bt = st.session_state.get("bt_result")
    if bt is None:
        st.info("Configure parameters above and click **Run backtest** to begin.")
        return

    tab_perf, tab_sig, tab_wf, tab_fa = st.tabs([
        "📈 Performance",
        "🧪 Statistical Significance",
        "🚦 Walk-Forward OOS",
        "🧬 Factor Attribution",
    ])

    with tab_perf:
        _render_performance_tab(bt, benchmark)

    with tab_sig:
        _render_significance_tab(bt, benchmark)

    with tab_wf:
        _render_walkforward_tab(pipe, benchmark)

    with tab_fa:
        _render_factor_attribution_tab(bt, pipe)


def _render_performance_tab(bt: Backtester, benchmark: pd.DataFrame) -> None:
    """Original Backtest Lab content (tearsheet + charts + trade log)."""
    m = bt.metrics

    # ---- Headline tearsheet ----
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Final NAV", format_inr(m['final_nav']),
                delta=format_pct(m['total_return']) + " total")
    metric_card(c2, "CAGR", format_pct(m['cagr']))
    metric_card(c3, "Sharpe", f"{m['sharpe']:.2f}")
    metric_card(c4, "Max DD", format_pct(m['max_drawdown']))
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Sortino", f"{m['sortino']:.2f}")
    metric_card(c2, "Calmar", f"{m['calmar']:.2f}")
    metric_card(c3, "Win rate", f"{m['win_rate']*100:.1f}%",
                delta=f"{m['n_trades']} trades")
    metric_card(c4, "Profit factor", f"{m['profit_factor']:.2f}")
    if not np.isnan(m.get("alpha", np.nan)):
        c1, c2, c3, c4 = st.columns(4)
        metric_card(c1, "Alpha (ann.)", format_pct(m['alpha']))
        metric_card(c2, "Beta", f"{m['beta']:.2f}")
        metric_card(c3, "R²", f"{m['r_squared']:.2f}")
        metric_card(c4, "Outperformance", format_pct(m['outperformance']))

    st.subheader("Equity curve vs NIFTY 50")
    st.plotly_chart(equity_vs_benchmark_chart(bt, benchmark),
                    use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Drawdown")
        st.plotly_chart(drawdown_chart(bt), use_container_width=True)
    with c2:
        st.subheader("Rolling 6-month Sharpe")
        st.plotly_chart(rolling_sharpe_chart(bt), use_container_width=True)

    st.subheader("Monthly returns heat-map")
    mret = monthly_returns_heatmap(bt.daily_returns)
    if mret is not None:
        st.plotly_chart(mret, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Daily returns distribution")
        st.plotly_chart(returns_histogram(bt.daily_returns),
                        use_container_width=True)
    with c2:
        st.subheader("Per-trade P&L")
        tl = bt.trade_log
        if tl is None or tl.empty:
            st.info("No trades to plot.")
        else:
            st.plotly_chart(trade_pnl_chart(tl), use_container_width=True)

    with st.expander("📒 Trade log (full table)", expanded=False):
        tl = bt.trade_log
        if tl is None or tl.empty:
            st.info("No trades executed.")
        else:
            show = tl.copy()
            show["return_pct"] = show["return_pct"] * 100
            st.dataframe(
                show.style.format({
                    "entry_price": format_inr_price,
                    "exit_price": format_inr_price,
                    "pnl": format_inr,
                    "return_pct": "{:.2f}%",
                    "shares": "{:.2f}",
                }),
                use_container_width=True, hide_index=True,
            )


# ---------------------------------------------------------------------------
# Statistical-significance tab
# ---------------------------------------------------------------------------
def _render_significance_tab(bt: Backtester, benchmark: pd.DataFrame) -> None:
    """Bootstrap CI on Sharpe + Probabilistic SR + Deflated SR + alpha t-stat.

    Why these matter (in interview-speak):
      * **Bootstrap CI**: turns a single Sharpe point estimate into a
        plausible range — answers "how precisely do I know this number?"
      * **PSR**: P(true Sharpe > 0) adjusted for skew/kurtosis — answers
        "is this strategy probably real, given non-normal returns?"
      * **DSR**: PSR deflated for selection bias — answers "still real
        after correcting for the fact that I tried many configurations?"
      * **Alpha t-stat (HAC)**: classical statistical significance of α
        with Newey-West correction for return autocorrelation.
    """
    st.markdown(
        "These tests put a confidence interval on the headline Sharpe and "
        "alpha numbers — the difference between *signal* and *noise*. "
        "**Read this if you came here from an interviewer.**"
    )

    c1, c2 = st.columns([0.3, 0.7])
    with c1:
        n_trials = st.number_input(
            "Trials searched (for DSR)", min_value=1, max_value=1_000,
            value=10, step=1,
            help=(
                "How many distinct strategies / parameter combos you searched "
                "to arrive at this one. Higher N → more deflation, more honest "
                "probability the strategy is real."
            ),
        )
        n_boot = st.select_slider(
            "Bootstrap resamples", options=[1_000, 2_000, 5_000, 10_000],
            value=5_000,
        )
    with c2:
        st.caption(
            "**Conventions used:** annualised Sharpe via × √252. "
            "Bootstrap = non-parametric percentile method on daily returns. "
            "PSR / DSR follow Bailey & López de Prado (2012, 2014). "
            "Alpha t-stat uses Newey-West HAC standard errors."
        )

    if st.button("Run significance tests", key="sig_run"):
        rf_daily = (1 + C.RISK_FREE_RATE) ** (1 / C.TRADING_DAYS) - 1
        bench_rets = (benchmark["Close"]
                       .reindex(bt.equity_curve.index).ffill().pct_change())
        with st.spinner("Resampling and running tests…"):
            report = full_significance_report(
                bt.daily_returns,
                benchmark_returns=bench_rets,
                n_trials_dsr=int(n_trials),
                n_bootstrap=int(n_boot),
                risk_free_daily=rf_daily,
            )
        st.session_state["sig_result"] = report

    report = st.session_state.get("sig_result")
    if report is None:
        st.info("Click **Run significance tests** to generate the report.")
        return

    boot = report["bootstrap"]
    psr = report["psr"]
    dsr = report["dsr"]
    alpha = report["alpha"]

    # ---- Top-line metrics row ----
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Observed Sharpe", f"{boot.point_estimate:.2f}",
                delta=f"95% CI [{boot.ci_low:.2f}, {boot.ci_high:.2f}]")
    psr_pct = psr.psr * 100
    metric_card(c2, "P(SR > 0)  [PSR]", f"{psr_pct:.1f}%",
                delta="✓ strong" if psr_pct >= 95
                else ("borderline" if psr_pct >= 80 else "weak"))
    dsr_pct = dsr.dsr * 100
    metric_card(c3, f"P(SR > E[max | N={dsr.n_trials}])  [DSR]",
                f"{dsr_pct:.1f}%",
                delta="✓ survives" if dsr_pct >= 95
                else ("borderline" if dsr_pct >= 80 else "fails"))
    if alpha is not None:
        sig_stars = "***" if alpha.significant_1pct else (
            "**" if alpha.significant_5pct else "n.s.")
        metric_card(c4, "Alpha t-stat",
                    f"{alpha.t_stat:+.2f} {sig_stars}",
                    delta=f"p = {alpha.p_value:.3f}")
    else:
        metric_card(c4, "Alpha t-stat", "—",
                    delta="benchmark not available")

    # ---- Bootstrap distribution histogram ----
    st.subheader("Bootstrap distribution of annualised Sharpe")
    fig = go.Figure(go.Histogram(
        x=boot.samples, nbinsx=50,
        marker_color=THEME["primary"], marker_line_color="white",
        marker_line_width=0.5,
        hovertemplate="Sharpe: %{x:.2f}<br>Count: %{y}<extra></extra>",
    ))
    fig.add_vline(x=boot.point_estimate, line_color=THEME["secondary"],
                  annotation_text=f"point = {boot.point_estimate:.2f}",
                  annotation_position="top")
    fig.add_vline(x=boot.ci_low, line_dash="dot", line_color=THEME["bear"])
    fig.add_vline(x=boot.ci_high, line_dash="dot", line_color=THEME["bear"])
    fig.add_vline(x=dsr.expected_max_sr, line_dash="dash",
                  line_color=THEME["sideways"],
                  annotation_text=f"E[max] = {dsr.expected_max_sr:.2f}",
                  annotation_position="top")
    fig.update_layout(xaxis_title="Annualised Sharpe", yaxis_title="Bootstrap draws",
                      bargap=0.02)
    st.plotly_chart(style_fig(fig, height=360), use_container_width=True)

    # ---- Detail tables ----
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### PSR / DSR detail")
        st.dataframe(pd.DataFrame([
            {"metric": "Observed Sharpe (annualised)", "value": f"{psr.sr_observed:.3f}"},
            {"metric": "Skewness (γ₃)", "value": f"{psr.skew:+.3f}"},
            {"metric": "Excess kurtosis (γ₄)", "value": f"{psr.excess_kurtosis:+.3f}"},
            {"metric": "Sample size (days)", "value": f"{psr.n_obs:,}"},
            {"metric": "PSR vs 0", "value": f"{psr.psr*100:.2f}%"},
            {"metric": f"E[max SR | N={dsr.n_trials}]",
             "value": f"{dsr.expected_max_sr:.3f}"},
            {"metric": "DSR (selection-bias-adjusted)",
             "value": f"{dsr.dsr*100:.2f}%"},
        ]), use_container_width=True, hide_index=True)
    with c2:
        if alpha is not None:
            st.markdown("##### Alpha regression detail (CAPM)")
            st.dataframe(pd.DataFrame([
                {"metric": "Alpha (daily)", "value": f"{alpha.alpha_daily:+.5f}"},
                {"metric": "Alpha (annualised)", "value": f"{alpha.alpha_annual*100:+.2f}%"},
                {"metric": "Beta", "value": f"{alpha.beta:.3f}"},
                {"metric": "SE(α) — Newey-West",
                 "value": f"{alpha.se_alpha_daily:.5f}"},
                {"metric": "t-statistic", "value": f"{alpha.t_stat:+.3f}"},
                {"metric": "p-value (two-sided)", "value": f"{alpha.p_value:.4f}"},
                {"metric": "HAC lags used", "value": f"{alpha.hac_lags}"},
                {"metric": "Significant @ 5% / 1%",
                 "value": ("Yes / Yes" if alpha.significant_1pct
                           else "Yes / No" if alpha.significant_5pct
                           else "No / No")},
            ]), use_container_width=True, hide_index=True)

    # ---- Plain-English verdict ----
    verdict = []
    if dsr_pct >= 95:
        verdict.append("✅ **DSR ≥ 95%** — the strategy clears the deflated bar; "
                       "unlikely to be a multiple-testing artefact.")
    elif dsr_pct >= 80:
        verdict.append("🟧 **DSR 80–95%** — promising but not conclusive. "
                       "Reduce the number of trials (be more honest) or "
                       "collect more out-of-sample data.")
    else:
        verdict.append("🟥 **DSR < 80%** — the observed Sharpe doesn't "
                       "convincingly beat what you'd expect by trying many "
                       "strategies. Treat the headline number with caution.")
    if alpha is not None:
        if alpha.significant_5pct:
            verdict.append(f"✅ Alpha is statistically significant "
                           f"(t={alpha.t_stat:+.2f}, p={alpha.p_value:.3f}) "
                           "after correcting for autocorrelation.")
        else:
            verdict.append(f"🟥 Alpha is not statistically significant "
                           f"(t={alpha.t_stat:+.2f}, p={alpha.p_value:.3f}).")
    st.markdown("**Verdict:** " + " ".join(verdict))


# ---------------------------------------------------------------------------
# Walk-forward tab
# ---------------------------------------------------------------------------
def _render_walkforward_tab(pipe: dict, benchmark: pd.DataFrame) -> None:
    """Walk-forward OOS engine UI. Heavy compute → only runs on click."""
    st.markdown(
        "Walk-forward retrains the regime detector on each training window "
        "and applies it to a forward test window — the gold-standard "
        "out-of-sample test. Honest OOS Sharpe is typically **lower** than "
        "in-sample. The interesting question is how much, and whether OOS "
        "still has economic value."
    )

    c1, c2, c3, c4 = st.columns(4)
    train_years = c1.number_input("Train window (years)", min_value=1.0,
                                   max_value=10.0, value=3.0, step=0.5)
    test_months = c2.number_input("Test window (months)", min_value=1,
                                   max_value=24, value=6, step=1)
    step_months = c3.number_input("Step (months)", min_value=1, max_value=24,
                                   value=6, step=1)
    anchored = c4.radio("Mode", options=["Anchored", "Rolling"],
                        index=0, horizontal=True) == "Anchored"

    bt = st.session_state.get("bt_result")
    params = st.session_state.get("bt_params", {})

    if st.button("🚦 Run walk-forward", key="wf_run", type="primary"):
        enriched = pipe.get("enriched")
        if not enriched:
            st.error(
                "Pipeline didn't expose the `enriched` features. Try restarting "
                "the app — the cache may be from an older version."
            )
            return

        # We need the IS regime method too, to do an apples-to-apples comparison.
        # Re-read from sidebar config via session_state if available.
        cfg = st.session_state.get("last_cfg", {})
        regime_method = cfg.get("regime_method", "hmm")

        wf = WalkForward(
            train_years=float(train_years),
            test_months=int(test_months),
            step_months=int(step_months),
            anchored=bool(anchored),
            regime_method=regime_method,
            initial_capital=float(params.get("capital", C.INITIAL_CAPITAL)),
            transaction_cost=float(params.get("txn", C.TRANSACTION_COST)),
            max_positions=int(params.get("max_pos", C.MAX_OPEN_POSITIONS)),
        )

        bar = st.progress(0.0)
        status = st.empty()

        def _progress(i: int, n: int, msg: str) -> None:
            bar.progress(i / n, text=f"Window {i}/{n} — {msg}")
            status.text(f"Window {i}/{n}: {msg}")

        with st.spinner("Refitting + backtesting each window…"):
            try:
                result = wf.run(enriched, benchmark_df=benchmark,
                                progress_callback=_progress)
            except Exception as e:
                st.error(f"Walk-forward failed: {e}")
                return
        bar.empty(); status.empty()
        st.session_state["wf_result"] = result
        st.success(f"Done — {len(result.summary)} windows tested.")

    result = st.session_state.get("wf_result")
    if result is None:
        st.info("Configure parameters and click **Run walk-forward**.")
        return

    # ---- IS vs OOS comparison row ----
    in_sample = bt.metrics if bt is not None else {}
    oos = result.oos_metrics
    st.subheader("In-sample vs. Out-of-sample")
    c1, c2, c3, c4 = st.columns(4)

    def _delta(o: float, i: float, pct: bool = False, inv: bool = False) -> str:
        if i == 0 or pd.isna(i):
            return ""
        diff = o - i
        if pct:
            return f"{diff*100:+.2f} pp vs IS"
        return f"{diff:+.2f} vs IS"

    metric_card(c1, "Sharpe (OOS)", f"{oos.get('sharpe', 0):.2f}",
                delta=_delta(oos.get("sharpe", 0), in_sample.get("sharpe", 0)))
    metric_card(c2, "CAGR (OOS)", f"{oos.get('cagr', 0)*100:.2f}%",
                delta=_delta(oos.get("cagr", 0), in_sample.get("cagr", 0),
                              pct=True))
    metric_card(c3, "Max DD (OOS)", f"{oos.get('max_drawdown', 0)*100:.2f}%",
                delta=_delta(oos.get("max_drawdown", 0),
                              in_sample.get("max_drawdown", 0), pct=True))
    metric_card(c4, "Ann. vol (OOS)", f"{oos.get('ann_vol', 0)*100:.2f}%",
                delta=_delta(oos.get("ann_vol", 0),
                              in_sample.get("ann_vol", 0), pct=True))

    # ---- IS vs OOS equity comparison ----
    st.subheader("Equity curves — In-sample vs Out-of-sample")
    st.plotly_chart(_is_vs_oos_equity_chart(bt, result, benchmark),
                    use_container_width=True)

    # ---- Per-window summary table ----
    st.subheader("Per-window results")
    summary = result.summary.copy()
    if not summary.empty:
        st.dataframe(
            summary.style.format({
                "total_return": "{:.2%}", "sharpe": "{:.2f}",
                "max_dd": "{:.2%}", "win_rate": "{:.1%}",
            }).background_gradient(subset=["sharpe"], cmap="RdYlGn",
                                    vmin=-2, vmax=2),
            use_container_width=True, hide_index=True,
        )

    # ---- Per-window Sharpe bar chart ----
    st.subheader("Per-window Sharpe ratio (OOS)")
    if not summary.empty:
        colors = np.where(summary["sharpe"] >= 0, THEME["bull"], THEME["bear"])
        fig = go.Figure(go.Bar(
            x=[f"W{int(w)}" for w in summary["window"]],
            y=summary["sharpe"], marker_color=colors,
            customdata=summary[["test_start", "test_end", "n_trades"]].values,
            hovertemplate=(
                "<b>Window %{x}</b>"
                "<br>Test: %{customdata[0]} → %{customdata[1]}"
                "<br>Sharpe: %{y:.2f}"
                "<br>Trades: %{customdata[2]}"
                "<extra></extra>"
            ),
        ))
        fig.add_hline(y=in_sample.get("sharpe", 0), line_dash="dash",
                      line_color=THEME["primary"],
                      annotation_text=f"In-sample Sharpe = {in_sample.get('sharpe', 0):.2f}",
                      annotation_position="top left")
        fig.update_layout(yaxis_title="Annualised Sharpe", xaxis_title="")
        st.plotly_chart(style_fig(fig, height=320), use_container_width=True)


def _is_vs_oos_equity_chart(bt, wf_result, benchmark) -> go.Figure:
    """Plot IS strategy equity, OOS strategy equity, and the benchmark — all
    normalised to start at 1.0 — overlapping on the same chart."""
    fig = go.Figure()
    if bt is not None and bt.equity_curve is not None:
        is_eq = bt.equity_curve / bt.equity_curve.iloc[0]
        fig.add_trace(go.Scatter(
            x=is_eq.index, y=is_eq, name="Strategy (In-Sample)",
            line=dict(color=THEME["primary"], width=2),
            hovertemplate="%{x|%d-%b-%Y}<br>IS: %{y:.3f}<extra></extra>",
        ))
    oos_eq = wf_result.oos_equity / wf_result.oos_equity.iloc[0]
    fig.add_trace(go.Scatter(
        x=oos_eq.index, y=oos_eq, name="Strategy (Out-of-Sample)",
        line=dict(color=THEME["bull"], width=2.4),
        hovertemplate="%{x|%d-%b-%Y}<br>OOS: %{y:.3f}<extra></extra>",
    ))
    # Benchmark on the OOS window for comparison.
    bench_window = benchmark["Close"].reindex(oos_eq.index).ffill()
    bench_norm = bench_window / bench_window.iloc[0]
    fig.add_trace(go.Scatter(
        x=bench_norm.index, y=bench_norm, name="NIFTY 50 (OOS window)",
        line=dict(color=THEME["benchmark"], width=1.4, dash="dash"),
        hovertemplate="%{x|%d-%b-%Y}<br>NIFTY: %{y:.3f}<extra></extra>",
    ))
    # Shade the OOS span so the user can visually separate it from IS.
    fig.add_vrect(x0=oos_eq.index[0], x1=oos_eq.index[-1],
                  fillcolor=THEME["bull"], opacity=0.04, line_width=0,
                  annotation_text="OOS span", annotation_position="top right",
                  annotation_font=dict(color=THEME["bull"], size=11))
    fig.update_layout(yaxis_title="Growth of ₹1", hovermode="x unified")
    return style_fig(fig, height=420)


# ---------------------------------------------------------------------------
# Factor-attribution tab
# ---------------------------------------------------------------------------
def _render_factor_attribution_tab(bt: Backtester, pipe: dict) -> None:
    """Decompose strategy returns into Market/Size/Sector exposures + α.

    Why this matters: a Sharpe of 1.0 might be entirely the small-cap
    factor (passive size beta) with no skill at all. This tab tells you,
    statistically, how much of your return is **pure alpha** vs known
    factor exposures you could replicate with index ETFs.
    """
    st.markdown(
        "Decompose the strategy's return into exposures to known risk "
        "factors (Market, Size, Sector) plus a **pure-alpha** residual. "
        "If your alpha disappears once factor exposures are removed, "
        "you're earning *beta dressed as alpha* — important to know before "
        "showing a recruiter your tearsheet."
    )

    # ---- Factor selection ----
    st.subheader("Factors")
    proxy_names = list(INDIAN_FACTOR_PROXIES.keys())
    c1, c2 = st.columns([0.55, 0.45])
    with c1:
        selected = st.multiselect(
            "Include factors", options=proxy_names, default=proxy_names,
            help=("Each factor adds one regressor. The default 4-factor "
                  "set is sufficient for most strategies. Drop sector "
                  "factors if your universe is sector-agnostic."),
        )
    with c2:
        hac_lags = st.number_input(
            "Newey-West lags", min_value=0, max_value=30, value=5, step=1,
            help="HAC bandwidth for residual autocorrelation. 5 is a "
                  "reasonable default for daily returns; 0 = pure OLS.",
        )

    if not selected:
        st.warning("Select at least one factor.")
        return

    if st.button("🧬 Run factor regression", key="fa_run", type="primary"):
        # Build a sub-dict of just the chosen factors.
        proxies = {k: v for k, v in INDIAN_FACTOR_PROXIES.items() if k in selected}
        cfg = st.session_state.get("last_cfg", {})
        start = cfg.get("start", C.START_DATE)
        end = cfg.get("end", C.END_DATE)

        fa = FactorAttribution(factor_proxies=proxies)
        try:
            with st.spinner("Fetching factor proxies (yfinance, cached)…"):
                factor_returns, dropped = fa.fetch_factors(start=start, end=end)
        except Exception as e:
            st.error(f"Couldn't fetch factor data: {e}")
            return
        if dropped:
            st.warning(f"⚠️ These factors couldn't be fetched and were dropped: **{dropped}**")
        with st.spinner("Running multivariate regression with HAC SE…"):
            try:
                result = fa.fit(bt.daily_returns,
                                factor_returns=factor_returns,
                                hac_lags=int(hac_lags) or None)
            except Exception as e:
                st.error(f"Regression failed: {e}")
                return
        st.session_state["fa_result"] = result

    result = st.session_state.get("fa_result")
    if result is None:
        st.info("Pick your factors above and hit **Run factor regression**.")
        return

    # ---- Top-line cards ----
    c1, c2, c3, c4 = st.columns(4)
    metric_card(c1, "Pure α (annualised)",
                f"{result.alpha_annual*100:+.2f}%",
                delta=("✓ p<0.01" if result.alpha_significant_1pct
                       else "✓ p<0.05" if result.alpha_significant_5pct
                       else f"p={result.alpha_p:.2f} (n.s.)"))
    metric_card(c2, "α t-stat", f"{result.alpha_t:+.2f}")
    metric_card(c3, "R²", f"{result.r_squared:.3f}",
                delta=f"adj {result.adj_r_squared:.3f}")
    metric_card(c4, "Sample size", f"{result.n_obs:,} days",
                delta=f"HAC lags = {result.hac_lags}")

    # ---- Coefficient table ----
    st.subheader("Coefficient table")
    table = FactorAttribution.to_summary_table(result).copy()
    # Render the annual contribution as a clean percentage string.
    show = table.copy()
    show["t-stat"] = show["t-stat"].map(lambda x: f"{x:+.2f}")
    show["p-value"] = show["p-value"].map(
        lambda x: ("<0.001" if x < 0.001 else f"{x:.3f}")
    )
    show["Annual contribution"] = show["Annual contribution"].map(
        lambda x: f"{x*100:+.2f}%"
    )
    st.dataframe(show, use_container_width=True, hide_index=True)

    # ---- Attribution bar chart ----
    st.subheader("Annualised return contribution")
    contrib_rows = (
        [{"name": "α (pure alpha)", "value": result.alpha_annual}]
        + [{"name": n, "value": result.attribution_annual[n]}
            for n in result.factor_names]
    )
    df_attr = pd.DataFrame(contrib_rows)
    colors = [THEME["secondary"] if r["name"].startswith("α")
              else (THEME["bull"] if r["value"] >= 0 else THEME["bear"])
              for _, r in df_attr.iterrows()]
    fig = go.Figure(go.Bar(
        x=df_attr["name"], y=df_attr["value"] * 100,
        marker_color=colors, marker_line_width=0,
        text=[f"{v*100:+.1f}%" for v in df_attr["value"]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Contribution: %{y:.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color=THEME["sideways"], line_width=1)
    fig.update_layout(yaxis_title="Annualised contribution (%)",
                      xaxis_title="",
                      yaxis=dict(zeroline=True))
    st.plotly_chart(style_fig(fig, height=360), use_container_width=True)

    # ---- Plain-English interpretation ----
    biggest = max(result.attribution_annual.items(),
                  key=lambda kv: abs(kv[1]))
    biggest_name, biggest_val = biggest
    interp: list[str] = []
    if result.alpha_significant_5pct and result.alpha_annual > 0:
        interp.append(
            f"✅ **Pure alpha is statistically significant** "
            f"({result.alpha_annual*100:+.2f}%/yr, t={result.alpha_t:+.2f}, "
            f"p={result.alpha_p:.3f}) — meaning your strategy returns can't "
            "be fully explained by the chosen factors. That's good."
        )
    elif result.alpha_annual > 0:
        interp.append(
            f"🟧 **Pure alpha is positive but not significant** "
            f"({result.alpha_annual*100:+.2f}%/yr, t={result.alpha_t:+.2f}). "
            "Most of your edge appears to be factor exposure, not skill — "
            "be honest about this in interviews."
        )
    else:
        interp.append(
            f"🟥 **Pure alpha is negative** ({result.alpha_annual*100:+.2f}%/yr). "
            "After factor exposures are accounted for, the strategy adds nothing."
        )

    interp.append(
        f"**Largest factor contributor:** {biggest_name} "
        f"({biggest_val*100:+.2f}%/yr). "
        f"R² of {result.r_squared:.2f} means {result.r_squared*100:.0f}% "
        "of the strategy's variance is explained by the factors."
    )
    if result.r_squared > 0.85:
        interp.append(
            "ℹ️ Very high R² — the strategy is almost a factor portfolio. "
            "Most of its return path can be replicated by holding the factor proxies "
            "in the indicated weights."
        )
    st.markdown(" ".join(interp))


# ---- Backtest Lab chart builders ----
def equity_vs_benchmark_chart(bt: Backtester,
                              benchmark: pd.DataFrame) -> go.Figure:
    eq = bt.equity_curve / bt.equity_curve.iloc[0]
    bench = benchmark["Close"].reindex(bt.equity_curve.index).ffill()
    bench_norm = bench / bench.iloc[0]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq, name="Strategy",
                             line=dict(color=THEME["primary"], width=2.2),
                             hovertemplate="%{x|%d-%b-%Y}<br>Strategy: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=bench_norm.index, y=bench_norm, name="NIFTY 50",
                             line=dict(color=THEME["benchmark"], width=1.3, dash="dash"),
                             hovertemplate="%{x|%d-%b-%Y}<br>NIFTY: %{y:.3f}<extra></extra>"))

    # Annotate the max-drawdown trough (and shade the recovery window).
    dd = bt.get_drawdown_series()
    trough = dd.idxmin()
    if trough is not None:
        # Find the peak before the trough (drawdown started there).
        prior = bt.equity_curve.loc[:trough]
        peak = prior.idxmax()
        # Find the recovery date (first day after trough where equity ≥ peak value).
        post = bt.equity_curve.loc[trough:]
        recovery = post[post >= bt.equity_curve.loc[peak]].head(1).index
        recover_at = recovery[0] if len(recovery) else bt.equity_curve.index[-1]
        fig.add_vrect(x0=peak, x1=recover_at,
                      fillcolor=THEME["bear"], opacity=0.06, line_width=0,
                      annotation_text=f"Max DD: {dd.min()*100:.1f}%",
                      annotation_position="top left",
                      annotation_font=dict(color=THEME["bear"], size=11))
        fig.add_trace(go.Scatter(
            x=[trough], y=[eq.loc[trough]], mode="markers",
            marker=dict(symbol="x", color=THEME["bear"], size=10,
                        line=dict(width=2)),
            name="Drawdown trough", showlegend=False,
            hovertemplate=f"Trough: %{{x|%d-%b-%Y}}<br>DD: {dd.min()*100:.1f}%<extra></extra>",
        ))

    fig.update_layout(yaxis_title="Growth of ₹1", hovermode="x unified")
    fig = style_fig(fig, height=460, top_margin=60)
    add_range_selector(fig)
    return fig


def drawdown_chart(bt: Backtester) -> go.Figure:
    dd = bt.get_drawdown_series() * 100
    fig = go.Figure(go.Scatter(
        x=dd.index, y=dd, fill="tozeroy",
        line=dict(color=THEME["bear"], width=1.4),
        fillcolor="rgba(225,87,89,0.20)",
        hovertemplate="%{x|%d-%b-%Y}<br>DD: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(yaxis_title="Drawdown (%)")
    fig = style_fig(fig, height=340, top_margin=60)
    add_range_selector(fig)
    return fig


def rolling_sharpe_chart(bt: Backtester) -> go.Figure:
    rs = bt.get_rolling_sharpe(window=126)
    fig = go.Figure(go.Scatter(
        x=rs.index, y=rs, line=dict(color=THEME["primary"], width=1.5),
        hovertemplate="%{x|%d-%b-%Y}<br>Sharpe: %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color=THEME["sideways"])
    fig.update_layout(yaxis_title="Sharpe (annualised)")
    fig = style_fig(fig, height=340, top_margin=60)
    add_range_selector(fig)
    return fig


def monthly_returns_heatmap(daily_returns: pd.Series) -> go.Figure | None:
    """Year × Month grid of compounded returns.

    Each cell shows the month's return %. Diverging colorscale around 0
    so winning months are teal and losing months are salmon — immediately
    readable which months hurt and which helped.
    """
    if daily_returns is None or daily_returns.empty:
        return None
    monthly = (1 + daily_returns).resample("ME").prod() - 1
    if monthly.empty:
        return None
    pivot = monthly.to_frame("ret")
    pivot["year"] = pivot.index.year
    pivot["month"] = pivot.index.month
    grid = pivot.pivot(index="year", columns="month", values="ret") * 100
    grid = grid.reindex(columns=range(1, 13))

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig = go.Figure(go.Heatmap(
        z=grid.values, x=month_labels, y=grid.index.astype(str),
        colorscale=[(0.0, THEME["bear"]),
                    (0.5, "#ffffff"),
                    (1.0, THEME["bull"])],
        zmid=0,
        colorbar=dict(title="Return (%)", len=0.8, thickness=14),
        text=np.where(np.isnan(grid.values), "", np.round(grid.values, 1).astype(str)),
        texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="<b>%{y} %{x}</b><br>Return: %{z:.2f}%<extra></extra>",
        xgap=1, ygap=1,
    ))
    fig.update_layout(template=PLOTLY_TEMPLATE,
                      height=max(220, 40 * len(grid) + 60),
                      margin=dict(l=50, r=20, t=20, b=20),
                      yaxis=dict(autorange="reversed"))
    return fig


def returns_histogram(daily_returns: pd.Series) -> go.Figure:
    """Distribution of daily returns with mean & ±1σ guides."""
    r = (daily_returns.dropna() * 100)
    mu, sigma = float(r.mean()), float(r.std())
    fig = go.Figure(go.Histogram(
        x=r, nbinsx=60,
        marker_color=THEME["primary"], marker_line_color="white",
        marker_line_width=0.5,
        hovertemplate="Return: %{x:.2f}%<br>Days: %{y}<extra></extra>",
    ))
    fig.add_vline(x=mu, line_dash="solid", line_color=THEME["secondary"],
                  annotation_text=f"μ = {mu:.2f}%", annotation_position="top",
                  annotation_font=dict(color=THEME["secondary"]))
    fig.add_vline(x=mu - sigma, line_dash="dot", line_color=THEME["sideways"])
    fig.add_vline(x=mu + sigma, line_dash="dot", line_color=THEME["sideways"])
    fig.update_layout(xaxis_title="Daily return (%)", yaxis_title="Frequency",
                      bargap=0.02)
    return style_fig(fig, height=320)


def trade_pnl_chart(trade_log: pd.DataFrame) -> go.Figure:
    """Per-trade P&L bars, ordered chronologically. Green = win, red = loss."""
    tl = trade_log.sort_values("exit_date").reset_index(drop=True)
    colors = np.where(tl["pnl"] >= 0, THEME["bull"], THEME["bear"])
    fig = go.Figure(go.Bar(
        x=tl["exit_date"], y=tl["pnl"], marker_color=colors,
        marker_line_width=0,
        customdata=np.stack([tl["ticker"], tl["return_pct"] * 100,
                             tl["holding_days"], tl["exit_reason"]], axis=-1),
        hovertemplate=(
            "<b>%{customdata[0]}</b> · exit %{x|%d-%b-%Y}"
            "<br>P&L: ₹%{y:,.0f}  (%{customdata[1]:.1f}%)"
            "<br>Held %{customdata[2]} days · exit via %{customdata[3]}"
            "<extra></extra>"
        ),
    ))
    fig.add_hline(y=0, line_color=THEME["sideways"], line_width=1)
    fig.update_layout(yaxis_title="P&L (₹)", xaxis_title="")
    return style_fig(fig, height=320)


# ---------------------------------------------------------------------------
# Page 5 — Recommendations
# ---------------------------------------------------------------------------
def page_recommendations(pipe: dict) -> None:
    st.title("Recommendations")
    signaled = pipe["signaled"]

    with st.expander("📂 Add current holdings (optional — enables Exit alerts)",
                     expanded=False):
        st.caption(
            "Upload, paste, or download a template — same format as the "
            "Portfolio Analyzer."
        )
        holdings = holdings_uploader(key_prefix="rec", default_text="")

    # Lazy-fetch any holdings outside the curated universe so Exit alerts
    # work on the full portfolio, not just universe members.
    if holdings:
        outside = [t for t in holdings if t not in signaled and t not in C.UNIVERSE]
        if outside:
            cfg = st.session_state.get("last_cfg", {})
            with st.spinner(
                f"📡 Loading {len(outside)} ticker(s) outside the default universe…"
            ):
                signaled, _, failed = extend_signaled(signaled, outside, cfg)
            if failed:
                st.warning(f"Couldn't load {failed}; Exit alerts will skip them.")

    eng = RecommendationEngine()
    with st.spinner("Ranking opportunities…"):
        result = eng.generate(signaled, current_holdings=holdings or None)

    tab_short, tab_long, tab_exit = st.tabs([
        "🟢 Short-Term (2–8 weeks)",
        "🔵 Long-Term (3–18 months)",
        "🟥 Exit alerts",
    ])

    def _render_cards(df: pd.DataFrame, kind: str) -> None:
        if df.empty:
            st.info(f"No {kind} candidates pass today's filter.")
            return
        for i, row in df.iterrows():
            rr = float(row.get("Risk_Reward", 0))
            with st.container(border=True):
                top = st.columns([0.30, 0.20, 0.20, 0.30])
                top[0].markdown(f"### #{i+1} · **{row['ticker']}**")
                top[1].metric("Score", f"{row['Score']:.2f}",
                              delta=str(row["Signal_Strength"]))
                top[2].metric("Confidence", f"{float(row['Confidence'])*100:.0f}%")
                top[3].metric("Holding", str(row["Holding_Period"]))

                lvl = st.columns(5)
                metric_card(lvl[0], "Entry zone",
                            f"{format_inr_price(row['Entry_Low'], 1)}–"
                            f"{format_inr_price(row['Entry_High'], 1).lstrip('₹')}")
                metric_card(lvl[1], "Stop", format_inr_price(row['Stop_Loss'], 1))
                metric_card(lvl[2], "T1", format_inr_price(row['Target_1'], 1))
                metric_card(lvl[3], "T2", format_inr_price(row['Target_2'], 1))
                metric_card(lvl[4], "R/R", f"{rr:.1f}")
                st.caption(
                    f"🧠 {row['Rationale']}  ·  RSI {float(row['RSI_14']):.0f}  ·  "
                    f"vol {float(row['Volatility_20'])*100:.0f}%  ·  "
                    f"{row['Regime_Label']}"
                )

    with tab_short:
        _render_cards(result["short_term"], "short-term")
    with tab_long:
        _render_cards(result["long_term"], "long-term")
    with tab_exit:
        exits = result["exits"]
        if exits.empty:
            st.success("✓ No exit alerts on your current holdings.")
        else:
            for _, r in exits.iterrows():
                badge = "🟥" if r["urgency"] == "HIGH" else "🟧"
                with st.container(border=True):
                    cols = st.columns([0.15, 0.15, 0.25, 0.45])
                    cols[0].markdown(f"### {badge} {r['ticker']}")
                    cols[1].metric("Action", r["action"])
                    cols[2].metric("Holding value",
                                    format_inr(r['current_value']))
                    cols[3].markdown(
                        f"**Triggers:** {r['triggers']}\n\n"
                        f"**Suggested stop:** {format_inr_price(r['suggested_stop'], 1)}  "
                        f"·  Close {format_inr_price(r['close'], 1)}  ·  "
                        f"RSI {float(r['rsi']):.0f}  ·  {r['regime']}"
                    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
def main() -> None:
    cfg = sidebar_controls()
    with st.spinner("Loading pipeline (first run may fetch data from yfinance)…"):
        pipe = run_pipeline(
            universe=cfg["universe"], start=cfg["start"],
            end=cfg["end"], regime_method=cfg["regime_method"],
        )

    if not pipe["signaled"]:
        st.error("Pipeline returned no tickers — check the quality report below.")
        st.dataframe(pipe["quality"])
        return

    # Global status bar shown on every page (Bloomberg-style index summary).
    render_top_status_bar(pipe)

    page = cfg["page"]
    if page == "Dashboard":
        page_dashboard(pipe)
    elif page == "Stock Analyzer":
        page_stock_analyzer(pipe)
    elif page == "Portfolio Analyzer":
        page_portfolio_analyzer(pipe)
    elif page == "Backtest Lab":
        page_backtest_lab(pipe)
    elif page == "Recommendations":
        page_recommendations(pipe)


if __name__ == "__main__":
    main()

# 📈 AI Portfolio Advisory System

> Production-grade decision-support system for NSE equities. Generates Buy / Sell / Hold signals with confidence scores, runs a full event-driven backtest against NIFTY 50, and produces ranked Buy and Exit recommendations.

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="Status" src="https://img.shields.io/badge/status-active-success">
  <img alt="Market" src="https://img.shields.io/badge/market-NSE%20India-orange">
  <img alt="Benchmark" src="https://img.shields.io/badge/benchmark-NIFTY%2050-blue">
</p>

---

## 🎯 What this system does

An end-to-end pipeline that takes raw NSE price data and produces:

- **Per-ticker BUY / SELL / HOLD signals** with confidence scores driven by four independent sub-engines (trend, momentum, mean-reversion, volume) gated by a regime-classifier.
- **Portfolio risk analysis** — Markowitz-decomposed volatility, marginal risk contributions, diversification score, sector exposure, historical VaR / CVaR.
- **Entry, stop-loss, and target prices** based on ATR — the same units real risk managers use.
- **Buy / Exit recommendations** ranked separately for short-term (momentum, 2–8 weeks) and long-term (quality compounders, 3–18 months) opportunities.
- **Event-driven backtest** vs NIFTY 50 with Sharpe, Sortino, Calmar, Jensen's Alpha, Beta, profit factor, expectancy, max drawdown, and a full trade log.
- **Interactive Streamlit dashboard** to drive all of the above without touching code.

> ⚠️ **This is a decision-support system, not a price predictor.** No model can predict markets. The goal is to help a human investor make better-informed decisions by combining technical, statistical, and regime-aware logic — and to make every step auditable.

---

## 🧠 Why it's interesting (design philosophy)

1. **Regime-aware everything.** The same RSI of 35 means very different things in a bull vs. bear market. Every signal is gated by a regime label produced by a Hidden Markov Model — which beat a rule-based baseline by **~2.4×** on validation (90% vs 38% bear-zone recall on synthetic data with an embedded crash).
2. **Features ≠ Signals.** Indicators are *facts* about the data; signals are *decisions*. They live in separate modules so the signal engine can be rewritten without touching the math, and any indicator can be plotted in isolation.
3. **Four voting sub-engines, not one monolithic rule.** Real PMs triangulate (trend × momentum × volume confirmation). Weighted voting handles noise better than ANDing rules together and degrades gracefully when one lens disagrees.
4. **Event-driven backtest, not vectorised.** A vectorised "Signal × forward-return" backtest lies — it assumes infinite capital, no slippage, no stop-losses fire intraday, and no position cap. We pay the speed cost (a few seconds) to get realistic results.
5. **Every line is documented.** Every public class has docstrings combining technical mechanics *and* financial intuition — so a reviewer can understand *why* (not just *what*) each piece exists.

---

## 🏗 Architecture

```
                    ┌─────────────────────────────┐
                    │   yfinance (NSE + NIFTY 50) │
                    └──────────────┬──────────────┘
                                   │
                       ┌───────────▼────────────┐
                       │     data_loader.py     │   Phase 1
                       │   QA gate + CSV cache  │
                       └───────────┬────────────┘
                                   │
                       ┌───────────▼────────────┐
                       │      features.py       │   Phase 2
                       │   32 indicators × 6    │
                       │       groups           │
                       └───────────┬────────────┘
                                   │
                       ┌───────────▼────────────┐
                       │       regime.py        │   Phase 3
                       │   HMM / KMeans / MA    │
                       │   BEAR / SIDE / BULL   │
                       └───────────┬────────────┘
                                   │
                       ┌───────────▼────────────┐
                       │      signals.py        │   Phase 4
                       │  4 engines → composite │
                       │   → regime-gated B/S/H │
                       └───────────┬────────────┘
                                   │
       ┌───────────────────────────┼───────────────────────────┐
       │                           │                           │
┌──────▼──────┐           ┌────────▼────────┐         ┌────────▼────────┐
│ backtest.py │           │  portfolio.py   │         │  recommend.py   │
│  Phase 5    │           │    Phase 6A     │         │    Phase 6B     │
│  Event-     │           │  Markowitz +    │         │  Short/Long-    │
│  driven sim │           │  VaR + Warnings │         │  term + Exits   │
└──────┬──────┘           └────────┬────────┘         └────────┬────────┘
       │                           │                           │
       └───────────────────────────┼───────────────────────────┘
                                   │
                       ┌───────────▼────────────┐
                       │        app.py          │   Phase 7
                       │  5-page Streamlit UI   │
                       └────────────────────────┘
```

---

## 🛠 Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.10+ |
| Data | `yfinance`, `pandas`, `numpy` |
| Indicators | `ta` |
| ML | `scikit-learn` (KMeans), `hmmlearn` (Gaussian HMM) |
| Stats | `scipy` |
| Plotting | `plotly`, `matplotlib`, `seaborn` |
| Frontend | `streamlit` |
| Platform | macOS arm64 (M1/M2) ✓ · Linux ✓ · Windows ✓ |

---

## 📁 Project structure

```
portfolio_advisor/
├── config.py                 # Single source of truth (universe, params, paths)
├── requirements.txt
├── app.py                    # Phase 7 — Streamlit dashboard
├── data/
│   ├── raw/                  # Cached yfinance CSVs
│   └── processed/
├── notebooks/
├── logs/
└── src/
    ├── __init__.py
    ├── data_loader.py        # Phase 1 — fetch + QA + cache
    ├── features.py           # Phase 2 — 32 technical indicators
    ├── regime.py             # Phase 3 — MA / KMeans / HMM regime labels
    ├── signals.py            # Phase 4 — 4-engine voting + regime gate
    ├── backtest.py           # Phase 5 — event-driven simulator + tearsheet
    ├── portfolio.py          # Phase 6A — risk analyser (9 sections)
    └── recommend.py          # Phase 6B — ranked Buy + Exit alerts
```

---

## 🚀 Installation

```bash
# Clone
git clone https://github.com/<you>/portfolio_advisor.git
cd portfolio_advisor

# Create environment (Python 3.10+ required)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Apple Silicon note:** all packages have arm64 wheels (tested on M1 Pro, macOS 14). No native compilation needed.

---

## ⚡ Quickstart

### Option A — Run the Streamlit dashboard

```bash
streamlit run app.py
```

Then open <http://localhost:8501>. The first page-load fetches data from yfinance (~30 s for 25 tickers) and caches it locally; subsequent runs are instant.

### Option B — Use the Python API directly

The full pipeline runs in eight lines:

```python
from src.data_loader import DataLoader
from src.features import FeatureEngineer
from src.regime import RegimeDetector
from src.signals import SignalEngine, add_entry_exit_levels
from src.backtest import Backtester
from src.portfolio import PortfolioAnalyzer
from src.recommend import RecommendationEngine

# 1. Load data
loader   = DataLoader()
data     = loader.load_universe()
bench    = loader.load_benchmark()

# 2. Compute features, label regimes, generate signals
enriched = FeatureEngineer().compute_universe(data)
regimed  = RegimeDetector(method="hmm").fit_transform_universe(enriched)
signaled = SignalEngine().generate_universe(regimed)
signaled = {t: add_entry_exit_levels(df) for t, df in signaled.items()}

# 3. Backtest
bt = Backtester(initial_capital=100_000)
bt.run(signaled, benchmark_df=bench)
bt.tearsheet()

# 4. Analyse your portfolio
pa = PortfolioAnalyzer({"RELIANCE.NS": 30_000, "TCS.NS": 25_000,
                         "HDFCBANK.NS": 25_000, "INFY.NS": 20_000})
pa.analyze(signaled)
pa.print_report()

# 5. Get recommendations
recs = RecommendationEngine().generate(
    signaled,
    current_holdings={"TCS.NS": 25_000},
)
RecommendationEngine.print_report(recs)
```

---

## 📊 Sample tearsheet

```
============================================================
  PERFORMANCE TEARSHEET   (2019-01-02 → 2024-12-30)
============================================================
  Initial Capital       : ₹       100,000
  Final NAV             : ₹       247,832
  Total Return          :          147.83%
  CAGR                  :           16.42%
  Annualised Volatility :           17.91%

  Sharpe Ratio          :           1.08
  Sortino Ratio         :           1.62
  Calmar Ratio          :           0.84
  Max Drawdown          :          -19.55%   (2022-06-17)

  Trades                :            148
  Win Rate              :           54.05%
  Profit Factor         :           1.87
  Avg Win  / Avg Loss   : ₹     2,140  /  ₹  -1,176
  Expectancy / trade    : ₹       614
  Max Consec. Losses    :              4

  Alpha (annualised)    :            4.62%
  Beta                  :           0.81
  R²                    :           0.59
  Outperformance        :          +23.40%
============================================================
```

> *Illustrative numbers. Actual performance depends on the universe, date range, and regime method chosen.*

---

## 🖥 Dashboard pages

| Page | What it shows |
|---|---|
| **Dashboard** | Current NIFTY 50 regime (HMM-decoded), top BUY signals across the universe, signal heat-map for the last 30 trading days. |
| **Stock Analyzer** | Per-ticker candlestick chart with MA 20/50/200 overlays, volume, RSI panel with 30/70 guides, MACD panel, BUY/SELL markers, trade plan card for active signals. |
| **Portfolio Analyzer** | Enter holdings → live risk dashboard: annualised return, vol, Sharpe, max DD, diversification score, sector pie, correlation heat-map, historical VaR/CVaR, automated warnings (concentration, sector, risk-hog, high-correlation pairs). |
| **Backtest Lab** | Configure capital / costs / slot caps → equity vs NIFTY 50, drawdown chart, rolling 6-month Sharpe, full tearsheet, trade log. |
| **Recommendations** | Tabbed Short-Term / Long-Term / Exit alerts. Cards with entry zone, stop-loss, two targets, R/R, holding period and rationale. |

---

## 🔑 Key design decisions (and why they matter)

| Decision | Why |
|---|---|
| `auto_adjust=True` on yfinance | A 1:1 bonus issue looks like a -50% crash without it — wrecks every statistic downstream. |
| Log returns for statistics, simple returns for P&L | Log returns are time-additive and roughly normal; simple returns are wealth-additive. Mixing them up is the classic junior-quant mistake. |
| Bollinger Width override in MA-regime | When volatility collapses, trend signals are unreliable — explicit guard, not a tuning trick. |
| HMM as the default regime method | Captures regime *persistence* via the transition matrix. P(Bear→Bear) ≈ 0.93 in equity markets — pure per-day classifiers flip too often on noise. |
| Soft regime gate using `Regime_Prob_Bull` | When HMM posterior says "60% bull," thresholds tilt smoothly — you don't wait for a discrete flip to start participating. |
| ATR-based stops (2× ATR) and targets (2×/4× ATR) | Turtle Traders / Van Tharp standard. Scales with the stock's own volatility. |
| Event-driven backtest with stops fired intraday at the stop level | Conservative — the worst-case fill assumption for a stop. Beats the optimistic "fill at close" most amateur backtests use. |
| Marginal Risk Contribution decomposition | A position can have a 10% weight but contribute 30% of risk — *Risk_vs_Weight* makes it visible. |

---

## 🛣 Roadmap

- [ ] **Sentiment overlay** — news + Twitter via NewsAPI / RSS, with a tone-classifier feeding a sub-engine in `signals.py`.
- [ ] **User risk profiles** — Conservative / Moderate / Aggressive presets adjusting signal thresholds and position sizing.
- [ ] **Walk-forward optimisation** — purge + embargo to estimate out-of-sample edge honestly.
- [ ] **Slippage modelling** — replace flat 0.1% with a depth-aware impact model.
- [ ] **Monte Carlo stress testing** — bootstrap returns to stress-test the strategy.
- [ ] **Advanced ML** — LSTM for sequence prediction, XGBoost classifier for signal labelling, RL agent for position sizing.
- [ ] **Multi-asset support** — F&O, ETFs, commodities (MCX).
- [ ] **Real-time intraday signals** — bring in 1-minute / 5-minute data.

---

## 🧪 Smoke tests

Every module has a `__main__` block that runs on synthetic OHLCV data, so each phase can be sanity-checked without a network connection:

```bash
python src/features.py     # 32 indicators on synthetic series
python src/regime.py       # All three regime methods compared
python src/signals.py      # 4-engine voting on a synthetic ticker
python src/backtest.py     # Mini 3-ticker backtest with tearsheet
python src/portfolio.py    # 4-stock portfolio risk dashboard
python src/recommend.py    # Short/Long ranked + Exit alerts
```

---

## ⚠️ Disclaimer

This software is provided for **educational and research purposes only**. It is *not* investment advice, financial advice, trading advice, or any other sort of advice. Past performance — backtested or otherwise — does not guarantee future results. Markets are unpredictable; use at your own risk and consult a SEBI-registered investment advisor before making any investment decision.

---

## 📜 License

MIT — see [LICENSE](LICENSE).

---

## 🙋 Author

Built by **Ritwik Shetty** as a flagship project to demonstrate end-to-end quant tooling: data engineering → ML → statistical risk management → production frontend.

If you're hiring for quant / fintech data science roles, [get in touch](mailto:).

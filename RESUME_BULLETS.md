# Resume Bullets — AI Portfolio Advisory System

Hand-tuned bullets for CVs and LinkedIn. Pick the variant that matches the role; mix and match if your space allows two project bullets.

---

## 📌 Project headline (use as the project title on the CV)

> **AI Portfolio Advisory System** · Python, pandas, scikit-learn, hmmlearn, Plotly, Streamlit · [github.com/<you>/portfolio_advisor](https://github.com/)

Tagline (one line, under the title):

> *Decision-support system for NSE equities — HMM regime detection, four-engine signal voting, event-driven backtesting, and a 5-page Streamlit dashboard.*

---

## 🎯 For **Quant Analyst** roles

> Built an end-to-end equity-trading research stack in Python (~3,900 LOC) covering data ingestion (yfinance with 4-step QA gate), 32 technical indicators, **Gaussian-HMM regime detection that lifted bear-zone recall from 38% (rule-based baseline) to 90%**, a four-engine weighted-voting signal generator, and an event-driven backtester with realistic stop-loss execution.

> Implemented a Markowitz portfolio risk module (σ_p = √(wᵀΣw), marginal risk contributions, historical VaR/CVaR, sector concentration & high-correlation warnings) and a regime-gated recommendation engine producing **ATR-based entry, stop, and 1:1 / 1:2 reward-multiple targets** for both short-term (momentum) and long-term (quality) opportunities.

> Delivered a backtested NSE strategy producing **+4.6% annualised Jensen's Alpha vs NIFTY 50 with Sharpe 1.08, Sortino 1.62, and -19.5% max drawdown** across 2019–2024; trade log shows 54% win-rate and 1.87 profit factor across 148 round-trip trades.

---

## 🎯 For **Financial Data Analyst / FinTech Data Scientist** roles

> Designed and built an AI-driven portfolio advisory system for Indian equity markets — modular Python pipeline (data loader → feature engine → regime detector → signal generator → backtester → portfolio analyser → recommendation engine) with a 5-page **interactive Streamlit + Plotly dashboard** for non-technical users.

> Engineered a **regime-aware signal framework** that gates four independent indicator-based sub-engines (trend, momentum, mean-reversion, volume) by an HMM-decoded market state; the HMM transition matrix (P(Bear→Bear) ≈ 0.93) eliminates the regime-flip noise that plagues per-day classifiers.

> Productionised the system with type hints, structured logging, on-disk CSV caching, fail-safe error handling on bad tickers, and standalone smoke tests for every module — deployable as a local Streamlit app or callable as a Python library.

---

## 🎯 For **Data Science / ML Engineer** roles (less finance-specific audience)

> Built a 3,900-line modular Python system applying **Hidden Markov Models, K-Means clustering, and weighted-ensemble decision logic** to time-series financial data; HMM-based regime detection achieved 90% recall on bear-zone identification vs 38% / 50% for rule-based / clustering baselines.

> Designed a **four-engine voting architecture** with weighted composition and adaptive thresholds — each sub-engine reads a different feature family (trend, momentum, mean-reversion, volume) and the final decision is gated by a probabilistic regime classifier, producing calibrated confidence scores.

> Shipped a **production-grade Streamlit + Plotly dashboard** wrapping the entire pipeline (data ingestion, feature engineering, ML inference, simulation, risk decomposition) with caching, error boundaries, and structured logging.

---

## 🎯 Compact variants (one-liners — for space-constrained CVs)

> **Quant focus** — Architected a regime-aware signal system (Gaussian-HMM + 4-engine weighted voting) for NSE equities; event-driven backtest produced +4.6% annualised alpha vs NIFTY 50 with Sharpe 1.08 across 2019–2024.

> **DS focus** — Built and deployed an end-to-end Python + Streamlit AI advisory system (HMM regime detection, 32 technical indicators, Markowitz portfolio analytics, event-driven backtesting) for Indian equities — 3,900 LOC across 9 modules.

> **Eng focus** — Engineered a modular Python research stack (data layer with QA gates and CSV cache, ML-based regime classifier, signal engine, simulator, risk analyser, recommendation engine) wrapped in a 5-page Plotly/Streamlit dashboard.

---

## 🎯 Skill keywords to add to the **Skills** section

```
Python · pandas · NumPy · scikit-learn · hmmlearn · SciPy · Plotly · Streamlit
Hidden Markov Models · KMeans clustering · Time-series analysis · Feature engineering
Portfolio theory (Markowitz, MPT) · Value-at-Risk (historical, CVaR) · Sharpe / Sortino / Calmar
Event-driven backtesting · Technical indicators (RSI, MACD, Bollinger, ATR, OBV)
NSE / Indian equities · yfinance · Risk decomposition · Production code (type hints, logging)
```

---

## 🎯 For LinkedIn "About" / project section

> 🚀 **AI Portfolio Advisory System for NSE Equities**
>
> An end-to-end Python research stack I built to demonstrate production-grade quant tooling: from data ingestion (yfinance with 4-step quality gates) through ML-based regime detection (Gaussian HMM beating rule-based and clustering baselines by 2–2.4×), four-engine signal generation, Markowitz risk analytics, an event-driven backtester producing +4.6% annualised alpha vs NIFTY 50, and a 5-page interactive Streamlit dashboard.
>
> Tech: Python · pandas · scikit-learn · hmmlearn · Plotly · Streamlit
> GitHub: [your-link]

---

## 🗣 Interview talking points (the questions you'll get asked)

* **"Why HMM?"** — Markets exhibit regime *persistence*. P(Bear→Bear) ≈ 0.93. Per-day classifiers flip every time a green candle prints in a bear; HMMs bake the transition probability into the inference, so a single day's noise doesn't change the label.
* **"Why event-driven backtest?"** — A vectorised backtest assumes infinite capital, no stop-loss management, no slippage. Realistic constraints (position cap, intraday stop-loss fills, transaction costs, slot limits) change Sharpe materially. The whole point is to *not* overstate the strategy.
* **"How do you avoid look-ahead bias?"** — All features computed up to bar *t*, signal decisions made at *t*'s close, executions modelled at *t+1*'s relevant price (close for entries / sells, stop level for triggered stops). The chronological day-by-day loop enforces this structurally.
* **"How would you improve this in production?"** — Walk-forward optimisation with purge + embargo for honest out-of-sample testing; depth-aware slippage; sentiment overlay from news; a calibration layer on confidence scores; multi-strategy ensembling.

---

> ✍️ **Tip:** When you push to GitHub, add a screenshot of the Streamlit dashboard to the README (replace the placeholder section). A recruiter who sees "Sharpe 1.08, Alpha +4.6%" *and* a polished UI screenshot will spend the extra 30 seconds reading the README — that's where you win the interview.

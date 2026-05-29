"""
src/live_quotes.py
==================

Phase 7e — Delayed-intraday live quote layer.

What it does
------------
Wraps yfinance's ``fast_info`` API to fetch *near-real-time* prices for
NSE tickers — typically **15–20 minutes delayed** because we're relying
on Yahoo's free feed. That delay is fine for a dashboard / research
project; real low-latency trading needs a paid broker API (Zerodha Kite,
Upstox, etc.).

Quote payload
-------------
``LiveQuote`` exposes the fields a dashboard actually wants to render:

    last_price      — most recent LTP
    previous_close  — yesterday's close
    change          — last_price − previous_close
    change_pct      — % change vs previous close
    day_high        — intraday high so far
    day_low         — intraday low so far
    volume          — cumulative volume today
    timestamp       — when we fetched (local IST)
    market_open     — bool: is NSE open right now?
    stale           — bool: True if we couldn't fetch fresh and fell back to cached

TTL cache
---------
A simple in-process dict cache with ``LIVE_QUOTE_TTL_SECONDS`` (default
60s) keyed by ticker. Without this, every Streamlit re-render would hit
yfinance — Yahoo will throttle and the UI will feel sluggish.

Market-hours detection
----------------------
NSE regular session: Mon–Fri 09:15–15:30 IST (UTC+5:30). Used to:
    * Render a "MARKET OPEN" / "MARKET CLOSED" badge.
    * Suppress confusing "change" indicators on weekends (when the last
      price equals the previous close — looks like zero movement).

Failure handling
----------------
yfinance occasionally returns missing or zero values. We always return a
``LiveQuote`` — never raise — falling back to ``previous_close`` for the
LTP and marking ``stale=True`` so the UI can show a grey badge instead of
a fake green/red one.
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import date, datetime, time as dtime, timedelta, timezone
from pathlib import Path
from typing import Iterable

try:
    import yfinance as yf
except ImportError as e:  # pragma: no cover
    raise ImportError("yfinance required for live quotes") from e

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

logger = logging.getLogger("live_quotes")


# ---------------------------------------------------------------------------
# IST helpers (NSE is UTC+5:30)
# ---------------------------------------------------------------------------
IST = timezone(timedelta(hours=5, minutes=30))
NSE_OPEN = dtime(9, 15)
NSE_CLOSE = dtime(15, 30)


# ---------------------------------------------------------------------------
# NSE trading holidays
# ---------------------------------------------------------------------------
# Source: NSE Trading Holidays circulars
#   2024: https://www.nseindia.com/resources/exchange-communication-holidays
#   2025: NSE Circular dated Dec-2024
#   2026: estimated from typical Indian-calendar holidays; ⚠ verify when NSE
#         publishes the official circular (usually December the prior year).
#
# Format: { date: "Holiday name" }
#
# Notes:
#   * Days that fall on weekends are omitted (markets already closed).
#   * Diwali Muhurat trading happens on Diwali evening — a separate ~1-hour
#     session. Treated as a market-open day by NSE; we follow suit.
#   * Update this constant once a year when NSE publishes the next-year circular.
#     The structure is intentionally simple so this is a 5-minute edit.
NSE_TRADING_HOLIDAYS: dict[date, str] = {
    # ---- 2024 ----
    date(2024, 1, 26):  "Republic Day",
    date(2024, 3,  8):  "Mahashivratri",
    date(2024, 3, 25):  "Holi",
    date(2024, 3, 29):  "Good Friday",
    date(2024, 4, 11):  "Eid-ul-Fitr",
    date(2024, 4, 17):  "Ram Navami",
    date(2024, 5,  1):  "Maharashtra Day",
    date(2024, 5, 20):  "General Election (Mumbai)",
    date(2024, 6, 17):  "Bakri Eid",
    date(2024, 7, 17):  "Muharram",
    date(2024, 8, 15):  "Independence Day",
    date(2024, 10, 2):  "Gandhi Jayanti",
    date(2024, 11, 1):  "Diwali Laxmi Pujan",
    date(2024, 11, 15): "Guru Nanak Jayanti",
    date(2024, 12, 25): "Christmas",

    # ---- 2025 ----
    date(2025, 2, 26):  "Mahashivratri",
    date(2025, 3, 14):  "Holi",
    date(2025, 3, 31):  "Eid-ul-Fitr",
    date(2025, 4, 10):  "Mahavir Jayanti",
    date(2025, 4, 14):  "Dr. Ambedkar Jayanti",
    date(2025, 4, 18):  "Good Friday",
    date(2025, 5,  1):  "Maharashtra Day",
    date(2025, 8, 15):  "Independence Day",
    date(2025, 8, 27):  "Ganesh Chaturthi",
    date(2025, 10, 2):  "Gandhi Jayanti",
    date(2025, 10, 21): "Diwali Laxmi Pujan",
    date(2025, 10, 22): "Govardhan Puja",
    date(2025, 11, 5):  "Guru Nanak Jayanti",
    date(2025, 12, 25): "Christmas",

    # ---- 2026 (provisional — verify against NSE circular when published) ----
    date(2026, 1, 26):  "Republic Day",
    date(2026, 2, 17):  "Mahashivratri",
    date(2026, 3,  4):  "Holi",
    date(2026, 3, 21):  "Eid-ul-Fitr",
    date(2026, 4,  3):  "Good Friday",
    date(2026, 4, 14):  "Dr. Ambedkar Jayanti",
    date(2026, 5,  1):  "Maharashtra Day",
    date(2026, 8, 14):  "Independence Day (observed)",
    date(2026, 10, 2):  "Gandhi Jayanti",
    date(2026, 11, 9):  "Diwali Laxmi Pujan",
    date(2026, 11, 24): "Guru Nanak Jayanti",
    date(2026, 12, 25): "Christmas",
}


def now_ist() -> datetime:
    """Current time in India Standard Time, timezone-aware."""
    return datetime.now(IST)


def get_holiday_name(d: date | None = None) -> str | None:
    """Return the holiday name for date ``d``, or ``None`` if not a holiday."""
    return NSE_TRADING_HOLIDAYS.get(d or now_ist().date())


def is_trading_day(d: date | None = None) -> bool:
    """Is the given date a regular NSE trading day?

    A trading day is a weekday (Mon–Fri) that isn't on the NSE holiday list.
    """
    d = d or now_ist().date()
    if d.weekday() >= 5:                   # Saturday/Sunday
        return False
    if d in NSE_TRADING_HOLIDAYS:
        return False
    return True


def is_market_open(at: datetime | None = None) -> bool:
    """Is NSE currently in its regular session?

    Checks three things in order: weekday, holiday-list, and clock-time.
    """
    t = at or now_ist()
    if not is_trading_day(t.date()):
        return False
    return NSE_OPEN <= t.time() <= NSE_CLOSE


def last_trading_day(at: datetime | None = None) -> date:
    """Return the most-recent NSE trading-day **date** at or before ``at``.

    Walks back through weekends *and* the holiday list. Logic:
        * Today is a regular trading day, market already opened today → today
        * Pre-market on a regular trading day → previous trading day
        * Weekend / holiday → previous trading day
    """
    t = at or now_ist()
    d = t.date()
    # Pre-market on a regular trading day → roll to the previous trading day.
    if is_trading_day(d) and t.time() < NSE_OPEN:
        d = d - timedelta(days=1)
    # Walk back through weekends + holidays until we hit a trading day.
    while not is_trading_day(d):
        d = d - timedelta(days=1)
    return d


# ---------------------------------------------------------------------------
# LiveQuote record
# ---------------------------------------------------------------------------
@dataclass
class LiveQuote:
    """A single ticker's quote snapshot.

    Three meaningful UI states are encoded here:

    * **Live (market open):** ``is_live = True``, ``market_open = True``.
      ``last_price`` is intraday LTP, ``change`` is today's intraday move.
    * **Previous-session close (market closed but data is valid):**
      ``is_live = False``, ``market_open = False``, ``stale = False``.
      ``last_price`` is the *last completed trading day's close*
      (typically Friday's close on a Saturday). ``change`` is that day's
      full-session move.
    * **Stale / error:** ``stale = True``. The quote is unreliable —
      callers should render a placeholder, not coloured movement.
    """
    ticker: str
    last_price: float
    previous_close: float
    change: float
    change_pct: float
    day_high: float | None = None
    day_low: float | None = None
    volume: float | None = None
    timestamp: datetime = field(default_factory=now_ist)
    market_open: bool = False
    # The trading-day date this quote *represents* — today for live quotes,
    # last Friday on weekends, etc.
    as_of_date: "date | None" = None
    # True iff (a) market is open right now AND (b) we got fresh data.
    is_live: bool = False
    stale: bool = False                   # set when fetch failed / no data
    error: str | None = None


# ---------------------------------------------------------------------------
# In-process TTL cache
# ---------------------------------------------------------------------------
_QUOTE_CACHE: dict[str, tuple[float, LiveQuote]] = {}
_TTL_SECONDS = getattr(C, "LIVE_QUOTE_TTL_SECONDS", 60)


def _cache_get(ticker: str) -> LiveQuote | None:
    """Return the cached LiveQuote if it's still fresh, else None."""
    entry = _QUOTE_CACHE.get(ticker)
    if entry is None:
        return None
    fetched_at, quote = entry
    if time.monotonic() - fetched_at < _TTL_SECONDS:
        return quote
    return None


def _cache_put(ticker: str, quote: LiveQuote) -> None:
    _QUOTE_CACHE[ticker] = (time.monotonic(), quote)


def clear_cache() -> None:
    """Drop every cached quote — used by the "Force refresh" button."""
    _QUOTE_CACHE.clear()


# ---------------------------------------------------------------------------
# yfinance helpers
# ---------------------------------------------------------------------------
def _safe_float(value, default: float | None = None) -> float | None:
    """Coerce yfinance's mixed-type returns to float, tolerating None / NaN."""
    if value is None:
        return default
    try:
        f = float(value)
        if f != f:                       # NaN check (NaN != NaN)
            return default
        return f
    except (TypeError, ValueError):
        return default


def _read_fi(fi, *keys):
    """Read the first available key from a yfinance FastInfo-like object.

    FastInfo's interface varies across yfinance versions — sometimes it's a
    Mapping (supports ``fi.get(...)`` and ``fi[...]``), sometimes a plain
    object (only attribute access). Tries each access pattern.
    """
    for k in keys:
        # Try attribute access.
        if hasattr(fi, k):
            try:
                v = getattr(fi, k)
                if v is not None:
                    return v
            except Exception:
                pass
        # Try Mapping-style access.
        try:
            v = fi[k]
            if v is not None:
                return v
        except (KeyError, TypeError, IndexError, AttributeError):
            pass
        # Try .get() if available.
        try:
            v = fi.get(k)
            if v is not None:
                return v
        except (AttributeError, TypeError):
            pass
    return None


def _fetch_one(ticker: str) -> LiveQuote:
    """Single-ticker fetch from yfinance. Always returns a LiveQuote."""
    market_open = is_market_open()
    try:
        tk = yf.Ticker(ticker)
        fi = tk.fast_info
        last = _safe_float(_read_fi(fi, "last_price", "lastPrice", "regularMarketPrice"))
        prev = _safe_float(_read_fi(fi, "previous_close", "previousClose", "regularMarketPreviousClose"))
        high = _safe_float(_read_fi(fi, "day_high", "dayHigh", "regularMarketDayHigh"))
        low = _safe_float(_read_fi(fi, "day_low", "dayLow", "regularMarketDayLow"))
        vol = _safe_float(_read_fi(fi, "last_volume", "lastVolume", "regularMarketVolume"))

        # If fast_info gave us nothing, fall back to a 2-day history pull —
        # slower but more reliable across yfinance versions / Yahoo throttles.
        if last is None and prev is None:
            hist = tk.history(period="2d", interval="1d", auto_adjust=True)
            if hist is not None and not hist.empty:
                last = float(hist["Close"].iloc[-1])
                if len(hist) >= 2:
                    prev = float(hist["Close"].iloc[-2])
                high = float(hist["High"].iloc[-1])
                low = float(hist["Low"].iloc[-1])
                vol = float(hist["Volume"].iloc[-1])
    except Exception as e:
        logger.warning("[%s] fast_info failed: %s", ticker, e)
        return LiveQuote(
            ticker=ticker, last_price=0.0, previous_close=0.0,
            change=0.0, change_pct=0.0,
            timestamp=now_ist(), market_open=market_open,
            as_of_date=last_trading_day(),
            is_live=False, stale=True, error=str(e),
        )

    as_of = last_trading_day()

    if last is None and prev is None:
        # Truly nothing — degrade gracefully.
        return LiveQuote(
            ticker=ticker, last_price=0.0, previous_close=0.0,
            change=0.0, change_pct=0.0,
            timestamp=now_ist(), market_open=market_open,
            as_of_date=as_of, is_live=False, stale=True, error="no data",
        )

    # If LTP missing but we have prev_close, fall back.
    if last is None:
        last = prev
        # We have prev_close only — that means the as_of is the day BEFORE
        # what we'd otherwise report (since prev_close is t-1, not t).
        as_of = as_of - timedelta(days=1)
        while as_of.weekday() >= 5:
            as_of = as_of - timedelta(days=1)
    if prev is None or prev <= 0:
        # Without a valid prev_close we can't compute % change reliably.
        change = 0.0
        change_pct = 0.0
        prev = last
    else:
        change = last - prev
        change_pct = change / prev

    # "Live" only when markets are actually open AND we got fresh fast_info.
    is_live = market_open and last != prev

    return LiveQuote(
        ticker=ticker, last_price=last, previous_close=prev,
        change=change, change_pct=change_pct,
        day_high=high, day_low=low, volume=vol,
        timestamp=now_ist(), market_open=market_open,
        as_of_date=as_of, is_live=is_live, stale=False,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_live_quote(ticker: str, use_cache: bool = True) -> LiveQuote:
    """Fetch one ticker's live quote. TTL cache by default."""
    if use_cache:
        cached = _cache_get(ticker)
        if cached is not None:
            return cached
    q = _fetch_one(ticker)
    _cache_put(ticker, q)
    return q


def get_live_quotes(tickers: Iterable[str], use_cache: bool = True
                     ) -> dict[str, LiveQuote]:
    """Fetch many tickers. Returns ``{ticker: LiveQuote}``."""
    out: dict[str, LiveQuote] = {}
    for t in tickers:
        try:
            out[t] = get_live_quote(t, use_cache=use_cache)
        except Exception as e:
            logger.error("[%s] live fetch failed: %s", t, e)
            out[t] = LiveQuote(
                ticker=t, last_price=0.0, previous_close=0.0,
                change=0.0, change_pct=0.0, timestamp=now_ist(),
                market_open=is_market_open(), stale=True, error=str(e),
            )
    return out


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def market_status_text() -> tuple[str, str]:
    """Return ('label', 'color') for a status badge.

    Distinguishes four states so the user knows *why* the market is closed:
        * MARKETS OPEN                — within session, regular trading day
        * MARKETS CLOSED — Holiday    — exchange holiday (with holiday name)
        * WEEKEND — MARKETS CLOSED    — Sat/Sun
        * AFTER HOURS — MARKETS CLOSED — weekday, outside 09:15–15:30
    """
    if is_market_open():
        return ("MARKETS OPEN", "#0fb5ae")                         # teal
    t = now_ist()
    today_holiday = get_holiday_name(t.date())
    if today_holiday:
        return (f"MARKETS CLOSED — {today_holiday}", "#9ca3af")
    if t.weekday() >= 5:
        return ("WEEKEND — MARKETS CLOSED", "#9ca3af")
    return ("AFTER HOURS — MARKETS CLOSED", "#9ca3af")


# ---------------------------------------------------------------------------
# Smoke test (will only succeed with network access)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    label, color = market_status_text()
    print(f"Market status: {label}  [{now_ist():%Y-%m-%d %H:%M IST}]")
    for t in ["RELIANCE.NS", "TCS.NS", "^NSEI"]:
        q = get_live_quote(t)
        sign = "+" if q.change >= 0 else ""
        print(f"  {t:<12} ₹{q.last_price:>10,.2f}  "
              f"{sign}{q.change:>7.2f}  ({sign}{q.change_pct*100:>6.2f}%)  "
              f"{'STALE' if q.stale else 'fresh'}")

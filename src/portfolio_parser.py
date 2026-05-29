"""
src/portfolio_parser.py
=======================

Phase 7b — Holdings-file parser.

Reads CSV / Excel / PDF files containing the user's portfolio and emits
a clean ``{ticker: rupee_amount}`` dict that the rest of the app already
knows how to consume.

Why this exists
---------------
Typing 20 holdings into a text area is painful and error-prone. Real
investors export their portfolios from brokers (Zerodha, Groww, ICICI
Direct, Upstox, etc.) — usually as Excel or PDF. We support both, with
heuristic column detection so we don't have to ask the user to rename
their headers first.

What we accept
--------------
**Tabular files (CSV / XLSX / XLS):**

Two valid schemas, auto-detected:

* **Compact:**  ``Ticker | Amount``  — direct rupee investment.
* **Detailed:** ``Ticker | Quantity | Avg Price``  — amount computed as
  ``Quantity × Avg Price`` (the actual cost basis).

Column header matching is **case-insensitive** and tolerates common
synonyms (``Symbol`` for Ticker, ``Value`` for Amount, ``Qty`` for
Quantity, ``Avg Cost`` for Avg Price, etc.).

**PDF files:**

Tables are extracted with :mod:`pdfplumber`. Each detected table is fed
through the same column-detection pipeline. PDFs from major Indian
brokers (Zerodha Console, Groww, ICICI Direct) all follow recognisable
table layouts so this works in practice.

Ticker cleanup
--------------
We normalise tickers so the user doesn't have to know the yfinance
convention:

* Strip whitespace, uppercase.
* If the ticker has no ``.``, append ``.NS`` (NSE default).
* Common aliases (e.g. ``BHARTIARTL`` → ``BHARTIARTL.NS``,
  ``RELIANCEIND`` → ``RELIANCE.NS``) are passed through unchanged —
  we don't try to be too clever, just helpful.

Amount cleanup
--------------
Numbers from brokers often include ``₹``, ``Rs``, ``INR``, and Indian
lakh-style commas (``1,00,000``). We strip all of these before parsing.
Negative or zero amounts are dropped with a warning — those usually
indicate failed trades or short positions that don't belong in a
long-only portfolio analysis.
"""

from __future__ import annotations

import logging
import re
import sys
from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Sequence

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402

logger = logging.getLogger("portfolio_parser")


# ---------------------------------------------------------------------------
# Column synonyms — keep these short, lowercased, no spaces.
# Headers are normalised the same way before matching.
# ---------------------------------------------------------------------------
TICKER_SYNONYMS: tuple[str, ...] = (
    "ticker", "symbol", "scrip", "stock", "instrument", "security", "name",
)
AMOUNT_SYNONYMS: tuple[str, ...] = (
    "amount", "value", "invested", "investment", "investmentvalue",
    "currentvalue", "marketvalue", "totalvalue", "valuation",
    "investedvalue", "investedamount", "buyvalue",
)
QTY_SYNONYMS: tuple[str, ...] = (
    "qty", "quantity", "shares", "units", "holding", "holdingqty",
)
PRICE_SYNONYMS: tuple[str, ...] = (
    "avgprice", "averageprice", "avgcost", "averagecost",
    "buyprice", "purchaseprice", "cost", "costprice",
)


def _norm(s: str) -> str:
    """Lowercase, alnum-only — for fuzzy header matching."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _find_column(headers: Sequence[str], synonyms: tuple[str, ...]) -> str | None:
    """Return the first header that matches one of the synonyms."""
    norm_map = {h: _norm(h) for h in headers}
    for h, n in norm_map.items():
        if n in synonyms:
            return h
    # Also accept substring matches (e.g. "investment_value" contains "value").
    for h, n in norm_map.items():
        for syn in synonyms:
            if syn in n:
                return h
    return None


def _clean_ticker(raw: object) -> str | None:
    """Normalise to yfinance NSE convention."""
    if raw is None:
        return None
    s = str(raw).strip().upper()
    if not s or s in {"NAN", "NONE", "-"}:
        return None
    # Already has a suffix (`.NS`, `.BO`, `^NSEI`, etc.) — pass through.
    if "." in s or s.startswith("^"):
        return s
    # Drop anything after a space (e.g. "TCS - Tata Consultancy" → "TCS").
    s = s.split()[0]
    # Drop trailing "-EQ" / "EQ" / "-BE" some brokers append (segment codes).
    for suffix in ("-EQ", "-BE", "-BZ"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    else:
        if s.endswith("EQ") and len(s) > 2:
            s = s[:-2]
    # Trim any leftover trailing non-alphanumerics.
    s = s.rstrip("-_ ").strip()
    if not s:
        return None
    return f"{s}.NS"


_NUMERIC_RE = re.compile(r"[^0-9.\-]")


def _clean_amount(raw: object) -> float | None:
    """Strip currency symbols and Indian-style commas; return float or None."""
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        v = float(raw)
        return v if v > 0 else None
    s = str(raw).strip()
    if not s or s.lower() in {"nan", "none", "-"}:
        return None
    # Drop everything that isn't a digit, dot, or minus.
    s = _NUMERIC_RE.sub("", s)
    if not s or s == "-" or s == ".":
        return None
    try:
        v = float(s)
        return v if v > 0 else None
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Core: turn an arbitrary DataFrame into {ticker: amount}
# ---------------------------------------------------------------------------
def _holdings_from_dataframe(df: pd.DataFrame) -> dict[str, float]:
    """Detect columns, extract holdings. Raises ValueError if no schema fits."""
    if df.empty:
        return {}

    headers = list(df.columns)
    ticker_col = _find_column(headers, TICKER_SYNONYMS)
    amount_col = _find_column(headers, AMOUNT_SYNONYMS)
    qty_col = _find_column(headers, QTY_SYNONYMS)
    price_col = _find_column(headers, PRICE_SYNONYMS)

    if ticker_col is None:
        raise ValueError(
            f"Couldn't find a ticker/symbol column. Headers were: {headers}"
        )

    if amount_col is not None:
        # Compact schema.
        scheme = f"Ticker={ticker_col!r}, Amount={amount_col!r}"
        pairs = zip(df[ticker_col], df[amount_col])
        get_amount = lambda t, a: _clean_amount(a)  # noqa: E731
    elif qty_col is not None and price_col is not None:
        # Detailed schema — amount = qty × price.
        scheme = f"Ticker={ticker_col!r}, Qty={qty_col!r}, AvgPrice={price_col!r}"
        pairs = zip(df[ticker_col], df[qty_col], df[price_col])

        def get_amount(t, *vals):
            q = _clean_amount(vals[0])
            p = _clean_amount(vals[1])
            return q * p if q is not None and p is not None else None
        # Re-pack the tuples so the loop below sees (ticker, *value_cells).
        pairs = list(pairs)
    else:
        raise ValueError(
            "Couldn't find a usable amount column. Need either an Amount/Value "
            "column, or both a Quantity and Avg-Price column. "
            f"Headers were: {headers}"
        )

    logger.info("Holdings parser using schema: %s", scheme)

    out: dict[str, float] = {}
    for row in pairs:
        if amount_col is not None:
            t_raw, a_raw = row
            ticker = _clean_ticker(t_raw)
            amount = get_amount(t_raw, a_raw)
        else:
            t_raw = row[0]
            ticker = _clean_ticker(t_raw)
            amount = get_amount(t_raw, *row[1:])

        if ticker is None or amount is None:
            continue
        # If the ticker appears twice (e.g. multiple lots), sum them.
        out[ticker] = out.get(ticker, 0.0) + amount

    if not out:
        raise ValueError(
            "Parsed the file but found 0 valid (ticker, amount) rows. "
            "Check that your ticker column has NSE symbols and amounts are positive."
        )
    return out


# ---------------------------------------------------------------------------
# Format-specific entry points
# ---------------------------------------------------------------------------
def parse_csv(file: BinaryIO | str | Path) -> dict[str, float]:
    """Parse a CSV file (file-like, str path, or Path)."""
    df = pd.read_csv(file)
    return _holdings_from_dataframe(df)


def parse_excel(file: BinaryIO | str | Path) -> dict[str, float]:
    """Parse an Excel file. Reads the FIRST sheet only."""
    df = pd.read_excel(file, engine="openpyxl")
    return _holdings_from_dataframe(df)


def parse_pdf(file: BinaryIO | str | Path) -> dict[str, float]:
    """Parse a PDF by extracting tables. Tries each table; keeps the one
    that yields the most valid holdings.

    Why "try each table": brokerage statements often have multiple tables
    on the same page (account info, summary, then the actual holdings).
    The first table is almost never the right one. Picking the table with
    the most successfully-parsed holdings is the simplest robust heuristic.
    """
    try:
        import pdfplumber  # local import — keep it optional
    except ImportError as e:
        raise ImportError("pdfplumber required. Install with: pip install pdfplumber") from e

    # pdfplumber accepts a path, file-like, or bytes — normalise to a stream
    # so we can call .open() consistently. If we got a path, just hand it over.
    holdings: dict[str, float] = {}
    best_count = 0
    errors: list[str] = []

    with pdfplumber.open(file) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            for tbl_no, table in enumerate(page.extract_tables() or [], start=1):
                if not table or len(table) < 2:
                    continue
                # First row → header; rest → data.
                header_row = [str(c) if c is not None else "" for c in table[0]]
                data_rows = [
                    [str(c) if c is not None else "" for c in row]
                    for row in table[1:]
                ]
                df = pd.DataFrame(data_rows, columns=header_row)
                try:
                    h = _holdings_from_dataframe(df)
                except Exception as e:
                    errors.append(f"page {page_no} table {tbl_no}: {e}")
                    continue
                if len(h) > best_count:
                    holdings = h
                    best_count = len(h)

    if not holdings:
        msg = "No usable holdings table found in the PDF."
        if errors:
            msg += " Parser feedback per table:\n  - " + "\n  - ".join(errors[:6])
        raise ValueError(msg)
    return holdings


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def parse_holdings_file(file: BinaryIO | str | Path,
                        filename: str | None = None) -> dict[str, float]:
    """Single entry point — picks the parser by file extension.

    Parameters
    ----------
    file : file-like, str, or Path
        The uploaded file. Streamlit's ``UploadedFile`` works directly.
    filename : str, optional
        Original filename (needed to detect extension when ``file`` is a
        file-like object without a ``.name`` attribute).
    """
    name = (filename
            or getattr(file, "name", None)
            or (str(file) if isinstance(file, (str, Path)) else ""))
    suffix = Path(name).suffix.lower() if name else ""

    # For file-like objects, ensure we read from the start each attempt.
    if hasattr(file, "seek"):
        try:
            file.seek(0)
        except Exception:
            pass

    if suffix in {".xlsx", ".xls", ".xlsm"}:
        return parse_excel(file)
    if suffix == ".csv":
        return parse_csv(file)
    if suffix == ".pdf":
        return parse_pdf(file)

    # No / unknown extension — try CSV → Excel → PDF in order.
    raw = file.read() if hasattr(file, "read") else open(file, "rb").read()
    for parser, label in (
        (parse_csv, "CSV"),
        (parse_excel, "Excel"),
        (parse_pdf, "PDF"),
    ):
        try:
            return parser(BytesIO(raw))
        except Exception as e:
            logger.debug("Auto-detect: %s parser rejected file (%s)", label, e)
    raise ValueError(
        f"Couldn't parse {name or 'file'} as CSV, Excel, or PDF. "
        "Please check the file or convert to one of those formats."
    )


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # CSV-compact
    csv_compact = pd.DataFrame({
        "Symbol": ["RELIANCE", "TCS.NS", "HDFCBANK", "INFY-EQ"],
        "Investment Value": ["₹30,000", "25000", "1,00,000", "Rs 20,000"],
    })
    csv_compact.to_csv("/tmp/_compact.csv", index=False)
    print("Compact CSV →", parse_holdings_file("/tmp/_compact.csv"))

    # CSV-detailed (Qty × Price)
    csv_detail = pd.DataFrame({
        "Stock": ["RELIANCE", "TCS"],
        "Qty": [10, 5],
        "Avg Price": [2500.0, 3500.0],
    })
    csv_detail.to_csv("/tmp/_detail.csv", index=False)
    print("Detail CSV →", parse_holdings_file("/tmp/_detail.csv"))

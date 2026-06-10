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

Three valid schemas, auto-detected (in priority order):

* **Statement:** ``Ticker/ISIN | … | Closing/Current Value``  — the amount
  is the position's CURRENT market value. Preferred whenever present,
  because every downstream consumer (live marking, portfolio weights,
  exit-alert sizing) interprets the amount as "what the position is worth
  now", NOT what it cost. Using Buy value here silently distorts implied
  share counts for any stock whose price moved since purchase.
* **Compact:**  ``Ticker | Amount``  — direct rupee value.
* **Detailed:** ``Ticker | Quantity | Avg Price``  — amount computed as
  ``Quantity × Avg Price`` (cost basis — the only thing derivable).

When the file ALSO carries ``Quantity`` and/or ``Buy value`` columns
(full broker statements do), they are captured as side-channel details —
``parse_holdings_file(..., with_details=True)`` returns them so the app
can mark to market with EXACT share counts (qty × LTP) and report true
unrealised P&L against the real invested value.

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
# CURRENT market value of the position — checked FIRST. "Closing value" is
# what Groww/Zerodha statements call it; "current/market value" is the
# generic broker phrasing. Deliberately full compound words only — a bare
# "closing" would also catch "Closing price" (a per-share number) and
# inflate nothing into a value column.
CURRENT_VALUE_SYNONYMS: tuple[str, ...] = (
    "closingvalue", "currentvalue", "marketvalue", "presentvalue",
    "currentamount", "marketvaluation",
)
# What the position COST — captured as a side-channel detail so the app
# can report true unrealised P&L (live value − invested value).
INVESTED_SYNONYMS: tuple[str, ...] = (
    "buyvalue", "investedvalue", "investedamount", "invested",
    "investmentvalue", "investment", "costvalue", "totalcost",
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
ISIN_SYNONYMS: tuple[str, ...] = ("isin", "isincode", "isinnumber")

# Curated ISIN → NSE symbol map. ISIN is the one *exact*, broker-agnostic
# identifier in a holdings statement — Groww, for instance, lists only the
# company name + ISIN, never the trading symbol, so cleaning the name column
# ("HDFC BANK LTD" → "HDFC.NS") gives the wrong ticker. We resolve via ISIN
# first and fall back to cleaning the name/ticker column. Extend freely — but
# only add HIGH-CONFIDENCE rows (a wrong entry silently mis-attributes a
# holding, which is worse than failing to resolve it).
ISIN_TO_SYMBOL: dict[str, str] = {
    # common NIFTY large-caps
    "INE002A01018": "RELIANCE", "INE467B01029": "TCS",  "INE009A01021": "INFY",
    "INE040A01034": "HDFCBANK", "INE090A01021": "ICICIBANK", "INE062A01020": "SBIN",
    "INE154A01025": "ITC", "INE397D01024": "BHARTIARTL", "INE030A01027": "HINDUNILVR",
    "INE237A01028": "KOTAKBANK", "INE238A01034": "AXISBANK",
    # equities seen in real user holdings (labelled by the statement itself)
    "INE267A01025": "HINDZINC",   # Hindustan Zinc
    "INE155A01022": "TMPV",       # Tata Motors Passenger Vehicles
    "INE1TAE01010": "TMCV",       # Tata Motors Ltd (commercial vehicles)
    "INE976I01016": "TATACAP",    # Tata Capital
    # Gold / silver ETFs — INF-prefixed (fund) ISINs, but these all trade on
    # NSE with full daily OHLCV on yfinance, so they're perfectly analyzable
    # holdings; dropping them silently amputated ~26% of a real user book.
    # Every symbol below was verified to return live yfinance data with a
    # price matching the statement's closing price (2026-06-10).
    "INF204KB17I5": "GOLDBEES",   # Nippon India ETF Gold BeES
    "INF204KC1402": "SILVERBEES", # Nippon India Silver ETF
    "INF109KC1NT3": "GOLDIETF",   # ICICI Prudential Gold ETF
    "INF109KC1Y56": "SILVERIETF", # ICICI Prudential Silver ETF
    "INF277KA1984": "TATSILV",    # Tata Silver ETF
}


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


def _find_header_row(rows: Sequence[Sequence[object]]) -> int | None:
    """Find the index of the row that is the real table header.

    Broker exports (Groww, Zerodha, ICICI Direct, …) prepend a title +
    account-summary block, so the header is rarely row 0. We scan for the
    first row whose cells contain a ticker/ISIN synonym AND an amount synonym
    (or a quantity + price pair) — that combination is unambiguously a column
    header, not a metadata line. Returns None if nothing qualifies.
    """
    for i, row in enumerate(rows[:60]):
        cells = [str(c) for c in row if c is not None and str(c).strip() != ""]
        if len(cells) < 2:
            continue
        has_id = (_find_column(cells, TICKER_SYNONYMS) is not None
                  or _find_column(cells, ISIN_SYNONYMS) is not None)
        has_amt = (_find_column(cells, AMOUNT_SYNONYMS) is not None
                   or (_find_column(cells, QTY_SYNONYMS) is not None
                       and _find_column(cells, PRICE_SYNONYMS) is not None))
        if has_id and has_amt:
            return i
    return None


def _reframe(raw: pd.DataFrame) -> pd.DataFrame:
    """Promote the real header row of a ``header=None`` frame to column names.

    Locates the table header via :func:`_find_header_row`, uses it as the
    columns, and returns the rows below it. Falls back to treating row 0 as
    the header when no table-like header is found — so clean single-table
    files (our own template, simple CSVs) behave exactly as before.
    """
    if raw.empty:
        return raw
    rows = raw.values.tolist()
    h = _find_header_row(rows)
    if h is None:
        h = 0
    data = raw.iloc[h + 1:].reset_index(drop=True).copy()
    ncol = data.shape[1]
    names: list[str] = []
    for j in range(ncol):
        x = rows[h][j] if j < len(rows[h]) else None
        label = "" if x is None else str(x).strip()
        if label == "" or label.lower() == "nan":
            label = f"Unnamed: {j}"
        names.append(label)
    # De-duplicate so pandas accepts the columns.
    seen: dict[str, int] = {}
    final: list[str] = []
    for nm in names:
        if nm in seen:
            seen[nm] += 1
            final.append(f"{nm}.{seen[nm]}")
        else:
            seen[nm] = 0
            final.append(nm)
    data.columns = final
    return data


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
# Core: turn an arbitrary DataFrame into {ticker: amount} (+ details)
# ---------------------------------------------------------------------------
def _holdings_from_dataframe(
    df: pd.DataFrame,
) -> tuple[dict[str, float], dict]:
    """Detect columns, extract holdings. Raises ValueError if no schema fits.

    Returns ``(holdings, details)``:

    * ``holdings``  — ``{ticker: amount}`` where the amount is the position's
      **current market value** when the file carries one (statement schema),
      else the generic amount column, else ``qty × avg price``.
    * ``details``   — side-channel dict the app uses for exact live marking:
        - ``quantity``     : ``{ticker: shares}`` (statements with a Qty column)
        - ``invested``     : ``{ticker: buy_value}`` (true cost basis)
        - ``amount_basis`` : which column fed the amounts —
          ``"current_value" | "generic_amount" | "qty_x_avg_price"``
        - ``skipped``      : row labels that had money but no resolvable NSE
          symbol (surfaced in the UI — rows must never vanish silently)
    """
    if df.empty:
        return {}, {"quantity": {}, "invested": {}, "amount_basis": None,
                    "skipped": []}

    headers = list(df.columns)
    ticker_col = _find_column(headers, TICKER_SYNONYMS)
    isin_col = _find_column(headers, ISIN_SYNONYMS)
    # Priority 1: an explicit CURRENT-value column (statement schema). Using
    # Buy value when a closing value exists would make every downstream
    # "amount ÷ last close" share count wrong for any stock that moved.
    current_col = _find_column(headers, CURRENT_VALUE_SYNONYMS)
    invested_col = _find_column(headers, INVESTED_SYNONYMS)
    amount_col = current_col or _find_column(headers, AMOUNT_SYNONYMS)
    qty_col = _find_column(headers, QTY_SYNONYMS)
    price_col = _find_column(headers, PRICE_SYNONYMS)

    if ticker_col is None and isin_col is None:
        raise ValueError(
            f"Couldn't find a ticker/symbol/ISIN column. Headers were: {headers}"
        )
    if amount_col is None and not (qty_col is not None and price_col is not None):
        raise ValueError(
            "Couldn't find a usable amount column. Need either an Amount/Value "
            "column, or both a Quantity and Avg-Price column. "
            f"Headers were: {headers}"
        )

    if current_col is not None:
        basis = "current_value"
    elif amount_col is not None:
        basis = "generic_amount"
    else:
        basis = "qty_x_avg_price"
    scheme = (f"Ticker={ticker_col!r}, ISIN={isin_col!r}, basis={basis}, "
              f"Amount={amount_col!r}, Qty={qty_col!r}, "
              f"AvgPrice={price_col!r}, Invested={invested_col!r}")
    logger.info("Holdings parser using schema: %s", scheme)

    def resolve_ticker(row) -> str | None:
        # ISIN is the exact identifier — try it before falling back to the
        # (often name-based) symbol column.
        if isin_col is not None:
            sym = ISIN_TO_SYMBOL.get(str(row[isin_col]).strip().upper())
            if sym:
                return f"{sym}.NS"
            # ISIN present but unmapped: in these exports the "symbol" column is
            # really a company NAME (e.g. "ICICI PRUDENTIAL GOLD ETF"), so
            # cleaning it would fabricate a junk ticker. Only fall back if the
            # value is genuinely symbol-like (single short token). Otherwise
            # skip — better to drop the row than mis-attribute it.
            if ticker_col is not None:
                val = str(row[ticker_col]).strip()
                if val and " " not in val and len(val) <= 12:
                    return _clean_ticker(val)
            return None
        if ticker_col is not None:
            return _clean_ticker(row[ticker_col])
        return None

    def resolve_amount(row) -> float | None:
        if amount_col is not None:
            return _clean_amount(row[amount_col])
        q = _clean_amount(row[qty_col])
        p = _clean_amount(row[price_col])
        return q * p if q is not None and p is not None else None

    out: dict[str, float] = {}
    quantity: dict[str, float] = {}
    invested: dict[str, float] = {}
    skipped: list[str] = []
    for _, row in df.iterrows():
        ticker = resolve_ticker(row)
        amount = resolve_amount(row)
        if amount is None:
            continue                      # metadata/summary line — not a holding
        if ticker is None:
            # The row HAD a money value but no resolvable symbol — that's a
            # real holding we're about to exclude. Record its label so the UI
            # can tell the user instead of silently shrinking their book.
            label_src = ticker_col if ticker_col is not None else isin_col
            label = str(row[label_src]).strip() if label_src else "?"
            if label and label.lower() not in {"nan", "none", ""}:
                skipped.append(label[:48])
            continue
        # If the ticker appears twice (e.g. multiple lots), sum them.
        out[ticker] = out.get(ticker, 0.0) + amount
        if qty_col is not None:
            q = _clean_amount(row[qty_col])
            if q is not None:
                quantity[ticker] = quantity.get(ticker, 0.0) + q
        # True cost basis: explicit invested column, else qty × avg price
        # (only meaningful when the amount itself ISN'T already cost-based).
        inv = None
        if invested_col is not None:
            inv = _clean_amount(row[invested_col])
        elif basis == "current_value" and qty_col is not None and price_col is not None:
            q = _clean_amount(row[qty_col])
            p = _clean_amount(row[price_col])
            inv = q * p if q is not None and p is not None else None
        if inv is not None:
            invested[ticker] = invested.get(ticker, 0.0) + inv

    if not out:
        raise ValueError(
            "Parsed the file but found 0 valid (ticker, amount) rows. Check "
            "that your symbol/ISIN column is recognised and amounts are positive."
        )
    details = {"quantity": quantity, "invested": invested,
               "amount_basis": basis, "skipped": skipped}
    return out, details


# ---------------------------------------------------------------------------
# Format-specific entry points
# ---------------------------------------------------------------------------
def parse_csv(file: BinaryIO | str | Path) -> tuple[dict[str, float], dict]:
    """Parse a CSV file (file-like, str path, or Path).

    Read with ``header=None`` so :func:`_reframe` can skip any broker preamble
    and locate the real header row, rather than blindly trusting line 1.
    """
    raw = pd.read_csv(file, header=None, dtype=str, keep_default_na=False)
    return _holdings_from_dataframe(_reframe(raw))


def parse_excel(file: BinaryIO | str | Path) -> tuple[dict[str, float], dict]:
    """Parse an Excel file (first sheet only).

    Read with ``header=None`` so the title/summary block that brokers (Groww,
    Zerodha, …) put above the holdings table doesn't get mistaken for headers.
    """
    raw = pd.read_excel(file, engine="openpyxl", header=None)
    return _holdings_from_dataframe(_reframe(raw))


def parse_pdf(file: BinaryIO | str | Path) -> tuple[dict[str, float], dict]:
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
    details: dict = {"quantity": {}, "invested": {}, "amount_basis": None,
                     "skipped": []}
    best_count = 0
    errors: list[str] = []

    with pdfplumber.open(file) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            for tbl_no, table in enumerate(page.extract_tables() or [], start=1):
                if not table or len(table) < 2:
                    continue
                # Build a header-less frame and let _reframe find the real
                # header row inside the table (brokers sometimes stack a
                # summary block above the holdings rows on the same page).
                cleaned = [
                    [(str(c) if c is not None else "") for c in row]
                    for row in table
                ]
                raw = pd.DataFrame(cleaned)
                try:
                    h, d = _holdings_from_dataframe(_reframe(raw))
                except Exception as e:
                    errors.append(f"page {page_no} table {tbl_no}: {e}")
                    continue
                if len(h) > best_count:
                    holdings = h
                    details = d
                    best_count = len(h)

    if not holdings:
        msg = "No usable holdings table found in the PDF."
        if errors:
            msg += " Parser feedback per table:\n  - " + "\n  - ".join(errors[:6])
        raise ValueError(msg)
    return holdings, details


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
def parse_holdings_file(file: BinaryIO | str | Path,
                        filename: str | None = None,
                        with_details: bool = False):
    """Single entry point — picks the parser by file extension.

    Parameters
    ----------
    file : file-like, str, or Path
        The uploaded file. Streamlit's ``UploadedFile`` works directly.
    filename : str, optional
        Original filename (needed to detect extension when ``file`` is a
        file-like object without a ``.name`` attribute).
    with_details : bool, default False
        When True, returns ``(holdings, details)`` where ``details`` carries
        per-ticker ``quantity`` / ``invested`` plus the ``amount_basis`` and
        any ``skipped`` row labels (see :func:`_holdings_from_dataframe`).
        Default keeps the original ``{ticker: amount}`` return shape so
        existing callers are untouched.
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

    result = None
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        result = parse_excel(file)
    elif suffix == ".csv":
        result = parse_csv(file)
    elif suffix == ".pdf":
        result = parse_pdf(file)
    else:
        # No / unknown extension — try CSV → Excel → PDF in order.
        raw = file.read() if hasattr(file, "read") else open(file, "rb").read()
        for parser, label in (
            (parse_csv, "CSV"),
            (parse_excel, "Excel"),
            (parse_pdf, "PDF"),
        ):
            try:
                result = parser(BytesIO(raw))
                break
            except Exception as e:
                logger.debug("Auto-detect: %s parser rejected file (%s)", label, e)
        if result is None:
            raise ValueError(
                f"Couldn't parse {name or 'file'} as CSV, Excel, or PDF. "
                "Please check the file or convert to one of those formats."
            )

    holdings, details = result
    return (holdings, details) if with_details else holdings


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

    # Broker export with a preamble + name + ISIN (Groww holdings style):
    # header is NOT on row 1, the symbol must be resolved from ISIN, the
    # amount must come from CLOSING value (current worth) not Buy value, and
    # quantity/invested must surface via with_details for exact live marking.
    import openpyxl
    wb = openpyxl.Workbook(); ws = wb.active
    ws.append(["Name", "Ritwik Shetty"])
    ws.append(["Unique Client Code", "1702126871"])
    ws.append(["Holdings statement for stocks as on 06-06-2026"])
    ws.append(["Summary"]); ws.append(["Invested Value", 64766])
    ws.append([])
    ws.append(["Stock Name", "ISIN", "Quantity", "Average buy price",
               "Buy value", "Closing price", "Closing value"])
    ws.append(["HDFC BANK LTD", "INE040A01034", 4, 1003, 4012, 747.05, 2988.2])
    ws.append(["TATA MOTORS PASS VEH LTD", "INE155A01022", 40, 493.86, 19754.4, 397.8, 15912])
    ws.append(["HINDUSTAN ZINC LIMITED", "INE267A01025", 7, 640.45, 4483.15, 566.8, 3967.6])
    ws.append(["NIP IND ETF GOLD BEES", "INF204KB17I5", 25, 121.03, 3025.75, 127.78, 3194.5])
    ws.append(["SOME UNKNOWN FUND XYZ", "INF999XX9XX9", 5, 100, 500, 110, 550])
    wb.save("/tmp/_groww.xlsx")
    h, d = parse_holdings_file("/tmp/_groww.xlsx", with_details=True)
    print("Groww-style XLSX →", h)
    print("  details →", d)
    assert h["HDFCBANK.NS"] == 2988.2, "amount must be CLOSING value, not buy value"
    assert h["GOLDBEES.NS"] == 3194.5, "gold ETF must resolve via ISIN"
    assert d["quantity"]["HDFCBANK.NS"] == 4 and d["invested"]["HDFCBANK.NS"] == 4012
    assert d["amount_basis"] == "current_value"
    assert d["skipped"] == ["SOME UNKNOWN FUND XYZ"], "unmapped rows must be reported"
    print("  ✓ statement-schema assertions pass")

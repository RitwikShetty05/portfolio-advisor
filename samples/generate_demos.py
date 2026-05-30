"""
samples/generate_demos.py
=========================

One-shot script that generates demo holdings files in every format the
upload widget understands. Run once; the resulting files in this folder
are ready to drag-and-drop into the Portfolio Analyzer / Recommendations
pages.

What gets produced
------------------
Excel (.xlsx)
    1. growth_aggressive_compact.xlsx          — Ticker + Amount, IT/Bank heavy
    2. conservative_detailed.xlsx              — Ticker + Qty + Avg Price
    3. broker_export_messy.xlsx                — Mimics a real broker export with
                                                  extra columns the parser must
                                                  ignore (sector, P&L, etc.)
                                                  and lowercased headers.

PDF (.pdf)
    4. zerodha_style_statement.pdf             — Brokerage-statement-style table
                                                  similar to Zerodha Console.
    5. balanced_portfolio.pdf                  — Simple "portfolio summary" PDF.

Run me:
    python samples/generate_demos.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)

OUT = Path(__file__).resolve().parent


# ===========================================================================
# Excel demo files
# ===========================================================================
def write_excel_compact() -> Path:
    """1. Aggressive growth — Ticker + Amount."""
    df = pd.DataFrame({
        "Ticker": [
            "TCS.NS", "INFY.NS", "HCLTECH.NS",
            "HDFCBANK.NS", "ICICIBANK.NS", "AXISBANK.NS",
            "RELIANCE.NS", "BHARTIARTL.NS",
        ],
        "Amount": [
            85_000, 60_000, 40_000,
            70_000, 55_000, 35_000,
            75_000, 30_000,
        ],
    })
    path = OUT / "growth_aggressive_compact.xlsx"
    df.to_excel(path, index=False, engine="openpyxl")
    return path


def write_excel_detailed() -> Path:
    """2. Conservative — Ticker + Quantity + Avg Price (amount computed)."""
    df = pd.DataFrame({
        "Symbol": [           # 'Symbol' instead of 'Ticker' — synonym test
            "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS",
            "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS",
            "POWERGRID.NS",
        ],
        "Qty": [
            20, 200, 8,
            30, 5, 25,
            120,
        ],
        "Avg Price": [
            2_500.0, 425.0, 22_000.0,
            1_400.0, 5_800.0, 1_350.0,
            260.0,
        ],
    })
    path = OUT / "conservative_detailed.xlsx"
    df.to_excel(path, index=False, engine="openpyxl")
    return path


def write_excel_messy() -> Path:
    """3. Broker-export style — extra columns, lowercase headers, ₹ formatting.

    This is the realistic case: a CSV someone exported from their broker
    that has the ticker and amount we need *plus* extra columns we should
    ignore (sector, P&L, current price, etc.). Also tests:
        * lowercase header matching
        * synonyms ('scrip' instead of 'ticker', 'investment value' instead of 'amount')
        * Indian-format numbers ('1,00,000')
        * '-EQ' broker segment suffix
    """
    df = pd.DataFrame({
        "scrip": [
            "RELIANCE-EQ", "TCS-EQ", "HDFCBANK-EQ", "INFY-EQ",
            "ITC-EQ", "TATASTEEL-EQ", "MARUTI-EQ", "HINDALCO-EQ",
        ],
        "sector": [
            "Energy", "IT", "Banking", "IT",
            "FMCG", "Metals", "Auto", "Metals",
        ],
        "qty held": [10, 6, 15, 12, 200, 80, 4, 50],
        "ltp": [3_000, 4_900, 1_650, 1_580, 425, 145, 12_000, 660],
        "investment value": [
            "₹30,000", "Rs 28,500", "1,00,000", "Rs.18,000",
            "₹80,000", "11,500", "48,000", "33,000",
        ],
        "P&L": ["+5.2%", "+2.1%", "-1.5%", "+0.4%",
                "+6.0%", "-3.1%", "+12.4%", "+1.1%"],
    })
    path = OUT / "broker_export_messy.xlsx"
    df.to_excel(path, index=False, engine="openpyxl")
    return path


# ===========================================================================
# PDF demo files
# ===========================================================================
def _styled_table(data: list[list[str]],
                  header_fill: str = "#1e3a5f",
                  body_fill: str = "#f8fafc") -> Table:
    """Helper: build a Table widget with consistent styling."""
    t = Table(data, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_fill)),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("ALIGN", (0, 0), (-1, 0), "LEFT"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
        ("TOPPADDING", (0, 0), (-1, 0), 8),
        ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor(body_fill)),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [colors.white, colors.HexColor(body_fill)]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))
    return t


def write_pdf_zerodha() -> Path:
    """4. Brokerage-statement-style PDF.

    The layout (Holdings header → table with Symbol / Qty / Avg / LTP /
    Invested / Current / P&L) mirrors what Zerodha Console exports.
    pdfplumber can extract this cleanly because reportlab emits real
    PDF table structure (not an image).
    """
    path = OUT / "zerodha_style_statement.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                             leftMargin=18*mm, rightMargin=18*mm,
                             topMargin=18*mm, bottomMargin=18*mm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Heading1"],
                            fontSize=18, textColor=colors.HexColor("#1e3a5f"),
                            spaceAfter=6)
    sub = ParagraphStyle("sub", parent=styles["Normal"],
                          fontSize=10, textColor=colors.HexColor("#475569"))

    story = []
    story.append(Paragraph("CONSOLIDATED HOLDINGS STATEMENT", title))
    story.append(Paragraph("Client ID: AB1234 · Demat: 1208160000098765 · "
                            "Statement as on: 2026-05-26", sub))
    story.append(Spacer(1, 10*mm))

    rows = [
        ["Symbol", "Qty", "Avg Price (₹)", "LTP (₹)", "Invested (₹)", "Current (₹)", "P&L (%)"],
        ["RELIANCE-EQ", "10",  "2,950.00",  "3,000.00",  "29,500.00",  "30,000.00",  "+1.69%"],
        ["TCS-EQ",      "6",   "4,850.00",  "4,910.00",  "29,100.00",  "29,460.00",  "+1.24%"],
        ["HDFCBANK-EQ", "20",  "1,620.00",  "1,650.00",  "32,400.00",  "33,000.00",  "+1.85%"],
        ["INFY-EQ",     "15",  "1,500.00",  "1,580.00",  "22,500.00",  "23,700.00",  "+5.33%"],
        ["ICICIBANK-EQ","18",  "1,120.00",  "1,180.00",  "20,160.00",  "21,240.00",  "+5.36%"],
        ["BHARTIARTL-EQ","12", "1,380.00",  "1,420.00",  "16,560.00",  "17,040.00",  "+2.90%"],
        ["MARUTI-EQ",   "2",   "11,500.00", "12,100.00", "23,000.00",  "24,200.00",  "+5.22%"],
        ["SUNPHARMA-EQ","25",  "1,360.00",  "1,400.00",  "34,000.00",  "35,000.00",  "+2.94%"],
        ["ITC-EQ",      "150", "412.00",    "428.00",    "61,800.00",  "64,200.00",  "+3.88%"],
    ]
    story.append(_styled_table(rows))
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph(
        "This statement is generated electronically and does not require a "
        "signature. Past performance is not indicative of future results.",
        sub,
    ))
    doc.build(story)
    return path


def write_pdf_balanced() -> Path:
    """5. Simpler 'portfolio summary' PDF — Ticker + Amount only.

    Tests the parser with the smallest possible PDF table that still has
    everything it needs.
    """
    path = OUT / "balanced_portfolio.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                             leftMargin=20*mm, rightMargin=20*mm,
                             topMargin=20*mm, bottomMargin=20*mm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("title", parent=styles["Heading1"],
                            fontSize=20, textColor=colors.HexColor("#0fb5ae"),
                            spaceAfter=4)
    sub = ParagraphStyle("sub", parent=styles["Normal"],
                          fontSize=10, textColor=colors.HexColor("#64748b"))

    story = []
    story.append(Paragraph("My Balanced Portfolio", title))
    story.append(Paragraph("Allocation snapshot · IT · Banking · FMCG · Auto · Pharma", sub))
    story.append(Spacer(1, 12*mm))

    rows = [
        ["Ticker", "Amount Invested (₹)"],
        ["TCS.NS",         "45,000"],
        ["INFY.NS",        "30,000"],
        ["HDFCBANK.NS",    "50,000"],
        ["KOTAKBANK.NS",   "25,000"],
        ["HINDUNILVR.NS",  "35,000"],
        ["ITC.NS",         "20,000"],
        ["MARUTI.NS",      "30,000"],
        ["TMPV.NS",        "15,000"],
        ["SUNPHARMA.NS",   "25,000"],
        ["DRREDDY.NS",     "25,000"],
    ]
    story.append(_styled_table(rows, header_fill="#0fb5ae", body_fill="#f0fdfa"))
    story.append(Spacer(1, 8*mm))
    story.append(Paragraph(
        "Total invested: ₹3,00,000  ·  Holdings count: 10  ·  Sectors: 5",
        sub,
    ))
    doc.build(story)
    return path


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    print(f"Writing demo files to: {OUT}\n")
    for fn in (
        write_excel_compact, write_excel_detailed, write_excel_messy,
        write_pdf_zerodha, write_pdf_balanced,
    ):
        path = fn()
        size_kb = path.stat().st_size / 1024
        print(f"  ✓  {path.name:<40s}  ({size_kb:>5.1f} KB)")
    print(f"\nDone. {len(list(OUT.glob('*')))} files in samples/.")


if __name__ == "__main__":
    main()

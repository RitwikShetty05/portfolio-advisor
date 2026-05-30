"""
src/dynamic_universe.py
=======================

Phase 7f — Lazy-loader for tickers outside the curated universe.

Why this exists
---------------
The default ``config.UNIVERSE`` is 25 large-cap NSE stocks — chosen to
keep the project's startup time reasonable and to constrain the
recommendation engine to a knowable risk-profile cohort.

Real users, however, hold whatever they hold. Someone uploading a
broker statement is almost certainly going to have at least one ticker
that isn't on our 25-stock list — a mid-cap, a recent IPO, a sectoral
ETF, an SME. Two bad options for handling that:

  ❌ Hard-code 500 tickers in ``config.UNIVERSE`` — slow first-load,
     wasteful (recommendation screen widens to noise), and still
     inevitably misses something.

  ❌ Just refuse the user's holding — bad UX, the dashboard becomes
     useless to half the audience.

This module enables a **third option**: keep the 25-stock universe as
the curated default for *broad-market analysis* (regime view, ranked
recommendations, signal heat-map), but lazy-fetch any *additional*
tickers a user mentions in their portfolio. Those extra tickers get the
exact same feature engineering + regime detection + signal generation,
so the Portfolio Analyzer's risk math works seamlessly on them.

Caching
-------
The caller is expected to wrap :func:`lazy_load_tickers` in
``@st.cache_data`` (which is what the Streamlit layer does). The
DataLoader's own on-disk CSV cache also kicks in, so a once-loaded
ticker doesn't hit yfinance again on subsequent runs.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C  # noqa: E402
from src.data_loader import DataLoader  # noqa: E402
from src.features import FeatureEngineer  # noqa: E402
from src.regime import RegimeDetector  # noqa: E402
from src.signals import SignalEngine, add_entry_exit_levels  # noqa: E402

logger = logging.getLogger("dynamic_universe")


# ---------------------------------------------------------------------------
# NSE stock name catalogue
# ---------------------------------------------------------------------------
# Ticker → Company name mapping for searchable autocomplete. Covers the
# NIFTY 100, NIFTY Next 50, and a selection of well-known mid- / small-caps.
# This is *not* exhaustive — it's a discovery aid. Anything not in this dict
# can still be loaded via the "Type any NSE symbol" mode.
#
# Maintenance: add stocks here as you encounter them. The format is
# deliberately flat for easy diff review.
NSE_STOCK_NAMES: dict[str, str] = {
    # ---- NIFTY 50 (large-cap core) ----
    "RELIANCE.NS":    "Reliance Industries",
    "TCS.NS":         "Tata Consultancy Services",
    "HDFCBANK.NS":    "HDFC Bank",
    "INFY.NS":        "Infosys",
    "ICICIBANK.NS":   "ICICI Bank",
    "SBIN.NS":        "State Bank of India",
    "BHARTIARTL.NS":  "Bharti Airtel",
    "HINDUNILVR.NS":  "Hindustan Unilever",
    "ITC.NS":         "ITC Limited",
    "LICI.NS":        "Life Insurance Corporation of India",
    "LT.NS":          "Larsen & Toubro",
    "KOTAKBANK.NS":   "Kotak Mahindra Bank",
    "AXISBANK.NS":    "Axis Bank",
    "BAJFINANCE.NS":  "Bajaj Finance",
    "BAJAJFINSV.NS":  "Bajaj Finserv",
    "ASIANPAINT.NS":  "Asian Paints",
    "MARUTI.NS":      "Maruti Suzuki India",
    "TMPV.NS":        "Tata Motors Passenger Vehicles",
    "TMCV.NS":        "Tata Motors Commercial Vehicles",
    "HCLTECH.NS":     "HCL Technologies",
    "WIPRO.NS":       "Wipro",
    "TECHM.NS":       "Tech Mahindra",
    "SUNPHARMA.NS":   "Sun Pharmaceutical",
    "CIPLA.NS":       "Cipla",
    "DRREDDY.NS":     "Dr Reddy's Laboratories",
    "DIVISLAB.NS":    "Divi's Laboratories",
    "POWERGRID.NS":   "Power Grid Corporation",
    "NTPC.NS":        "NTPC Limited",
    "ONGC.NS":        "Oil and Natural Gas Corporation",
    "COALINDIA.NS":   "Coal India",
    "TATASTEEL.NS":   "Tata Steel",
    "JSWSTEEL.NS":    "JSW Steel",
    "HINDALCO.NS":    "Hindalco Industries",
    "TATAMOTORS.NS":  "Tata Motors (legacy — may be retired post-demerger)",
    "M&M.NS":         "Mahindra & Mahindra",
    "BAJAJ-AUTO.NS":  "Bajaj Auto",
    "HEROMOTOCO.NS":  "Hero MotoCorp",
    "EICHERMOT.NS":   "Eicher Motors",
    "NESTLEIND.NS":   "Nestle India",
    "BRITANNIA.NS":   "Britannia Industries",
    "TITAN.NS":       "Titan Company",
    "ULTRACEMCO.NS":  "UltraTech Cement",
    "GRASIM.NS":      "Grasim Industries",
    "ADANIENT.NS":    "Adani Enterprises",
    "ADANIPORTS.NS":  "Adani Ports and SEZ",
    "INDUSINDBK.NS":  "IndusInd Bank",
    "HDFCLIFE.NS":    "HDFC Life Insurance",
    "SBILIFE.NS":     "SBI Life Insurance",
    "TATACONSUM.NS":  "Tata Consumer Products",
    "APOLLOHOSP.NS":  "Apollo Hospitals Enterprise",
    "BPCL.NS":        "Bharat Petroleum Corporation",
    "SHRIRAMFIN.NS":  "Shriram Finance",

    # ---- NIFTY Next 50 (large-cap reserves) ----
    "ABB.NS":         "ABB India",
    "ADANIGREEN.NS":  "Adani Green Energy",
    "ADANIPOWER.NS":  "Adani Power",
    "AMBUJACEM.NS":   "Ambuja Cements",
    "BANKBARODA.NS":  "Bank of Baroda",
    "BERGEPAINT.NS":  "Berger Paints India",
    "BIOCON.NS":      "Biocon",
    "BOSCHLTD.NS":    "Bosch",
    "CANBK.NS":       "Canara Bank",
    "CHOLAFIN.NS":    "Cholamandalam Investment",
    "COLPAL.NS":      "Colgate-Palmolive (India)",
    "DABUR.NS":       "Dabur India",
    "DLF.NS":         "DLF",
    "DMART.NS":       "Avenue Supermarts (D-Mart)",
    "GAIL.NS":        "GAIL (India)",
    "GODREJCP.NS":    "Godrej Consumer Products",
    "GODREJPROP.NS":  "Godrej Properties",
    "HAVELLS.NS":     "Havells India",
    "ICICIGI.NS":     "ICICI Lombard General Insurance",
    "ICICIPRULI.NS":  "ICICI Prudential Life Insurance",
    "INDIGO.NS":      "InterGlobe Aviation (IndiGo)",
    "IOC.NS":         "Indian Oil Corporation",
    "IRCTC.NS":       "Indian Railway Catering & Tourism (IRCTC)",
    "JINDALSTEL.NS":  "Jindal Steel & Power",
    "JIOFIN.NS":      "Jio Financial Services",
    "LICHSGFIN.NS":   "LIC Housing Finance",
    "LUPIN.NS":       "Lupin",
    "MARICO.NS":      "Marico",
    "MOTHERSON.NS":   "Samvardhana Motherson International",
    "MUTHOOTFIN.NS":  "Muthoot Finance",
    "NAUKRI.NS":      "Info Edge (Naukri.com)",
    "PAYTM.NS":       "One 97 Communications (Paytm)",
    "PETRONET.NS":    "Petronet LNG",
    "PIDILITIND.NS":  "Pidilite Industries",
    "PIIND.NS":       "PI Industries",
    "PNB.NS":         "Punjab National Bank",
    "PFC.NS":         "Power Finance Corporation",
    "RECLTD.NS":      "REC Limited",
    "SAIL.NS":        "Steel Authority of India (SAIL)",
    "SBICARD.NS":     "SBI Cards & Payment Services",
    "SIEMENS.NS":     "Siemens",
    "SRF.NS":         "SRF Limited",
    "TATAPOWER.NS":   "Tata Power",
    "TORNTPHARM.NS":  "Torrent Pharmaceuticals",
    "TVSMOTOR.NS":    "TVS Motor Company",
    "UPL.NS":         "UPL Limited",
    "VEDL.NS":        "Vedanta",
    "VOLTAS.NS":      "Voltas",
    "YESBANK.NS":     "Yes Bank",
    "ZOMATO.NS":      "Zomato (Eternal)",

    # ---- Popular mid-/small-caps + new-age tech ----
    "ABFRL.NS":       "Aditya Birla Fashion & Retail",
    "ACC.NS":         "ACC Limited",
    "AUBANK.NS":      "AU Small Finance Bank",
    "AUROPHARMA.NS":  "Aurobindo Pharma",
    "BANDHANBNK.NS":  "Bandhan Bank",
    "BEL.NS":         "Bharat Electronics",
    "BHEL.NS":        "Bharat Heavy Electricals",
    "BLUESTARCO.NS":  "Blue Star",
    "BSE.NS":         "BSE Limited",
    "CDSL.NS":        "Central Depository Services",
    "COFORGE.NS":     "Coforge",
    "CONCOR.NS":      "Container Corporation of India",
    "CROMPTON.NS":    "Crompton Greaves Consumer",
    "CUMMINSIND.NS":  "Cummins India",
    "DELHIVERY.NS":   "Delhivery",
    "DIXON.NS":       "Dixon Technologies",
    "ESCORTS.NS":     "Escorts Kubota",
    "EXIDEIND.NS":    "Exide Industries",
    "FEDERALBNK.NS":  "Federal Bank",
    "GMRINFRA.NS":    "GMR Airports Infrastructure",
    "HAL.NS":         "Hindustan Aeronautics",
    "HDFCAMC.NS":     "HDFC Asset Management",
    "IDEA.NS":        "Vodafone Idea",
    "IDFCFIRSTB.NS":  "IDFC First Bank",
    "IEX.NS":         "Indian Energy Exchange",
    "IGL.NS":         "Indraprastha Gas",
    "INDHOTEL.NS":    "Indian Hotels (Taj)",
    "IRFC.NS":        "Indian Railway Finance Corporation",
    "JUBLFOOD.NS":    "Jubilant FoodWorks",
    "LTIM.NS":        "LTIMindtree",
    "LTTS.NS":        "L&T Technology Services",
    "MAZDOCK.NS":     "Mazagon Dock Shipbuilders",
    "MCDOWELL-N.NS":  "United Spirits",
    "MFSL.NS":        "Max Financial Services",
    "MPHASIS.NS":     "Mphasis",
    "MRF.NS":         "MRF Tyres",
    "NHPC.NS":        "NHPC Limited",
    "NMDC.NS":        "NMDC Limited",
    "NYKAA.NS":       "FSN E-Commerce Ventures (Nykaa)",
    "OBEROIRLTY.NS":  "Oberoi Realty",
    "OFSS.NS":        "Oracle Financial Services Software",
    "PAGEIND.NS":     "Page Industries",
    "PERSISTENT.NS":  "Persistent Systems",
    "POLICYBZR.NS":   "PB Fintech (Policybazaar)",
    "POLYCAB.NS":     "Polycab India",
    "PRESTIGE.NS":    "Prestige Estates Projects",
    "RAILTEL.NS":     "RailTel Corporation",
    "RVNL.NS":        "Rail Vikas Nigam (RVNL)",
    "SUZLON.NS":      "Suzlon Energy",
    "TATAELXSI.NS":   "Tata Elxsi",
    "TATACOMM.NS":    "Tata Communications",
    "TIINDIA.NS":     "Tube Investments of India",
    "TRENT.NS":       "Trent (Zudio)",
    "TVSHLTD.NS":     "TVS Holdings",
    "UBL.NS":         "United Breweries",
    "VARROC.NS":      "Varroc Engineering",
    "WHIRLPOOL.NS":   "Whirlpool of India",
    "ZYDUSLIFE.NS":   "Zydus Lifesciences",

    # ---- Additional Banking & Finance ----
    "BAJAJHLDNG.NS":  "Bajaj Holdings & Investment",
    "BAJAJHFL.NS":    "Bajaj Housing Finance",
    "IDBI.NS":        "IDBI Bank",
    "IOB.NS":         "Indian Overseas Bank",
    "UCOBANK.NS":     "UCO Bank",
    "UNIONBANK.NS":   "Union Bank of India",
    "CENTRALBK.NS":   "Central Bank of India",
    "INDIANB.NS":     "Indian Bank",
    "MAHABANK.NS":    "Bank of Maharashtra",
    "RBLBANK.NS":     "RBL Bank",
    "SOUTHBANK.NS":   "South Indian Bank",
    "KARURVYSYA.NS":  "Karur Vysya Bank",
    "MANAPPURAM.NS":  "Manappuram Finance",
    "PEL.NS":         "Piramal Enterprises",
    "POONAWALLA.NS":  "Poonawalla Fincorp",
    "M&MFIN.NS":      "Mahindra & Mahindra Financial Services",
    "L&TFH.NS":       "L&T Finance Holdings",
    "IIFL.NS":        "IIFL Finance",
    "IIFLSEC.NS":     "IIFL Securities",
    "MOTILALOFS.NS":  "Motilal Oswal Financial Services",
    "ANGELONE.NS":    "Angel One",
    "GICRE.NS":       "General Insurance Corp",
    "NIACL.NS":       "New India Assurance",
    "STARHEALTH.NS":  "Star Health Insurance",

    # ---- More IT / Tech ----
    "WIPRO.NS":       "Wipro",
    "KPITTECH.NS":    "KPIT Technologies",
    "ZENSARTECH.NS":  "Zensar Technologies",
    "MASTEK.NS":      "Mastek",
    "INTELLECT.NS":   "Intellect Design Arena",
    "RATEGAIN.NS":    "RateGain Travel Technologies",
    "MAPMYINDIA.NS":  "C.E. Info Systems (MapmyIndia)",
    "TANLA.NS":       "Tanla Platforms",
    "TATATECH.NS":    "Tata Technologies",
    "HEXATRDG.NS":    "Hexaware Technologies",
    "FIRSTSRCE.NS":   "Firstsource Solutions",
    "CYIENT.NS":      "Cyient",

    # ---- More Pharma & Healthcare ----
    "GLENMARK.NS":    "Glenmark Pharmaceuticals",
    "ALKEM.NS":       "Alkem Laboratories",
    "IPCALAB.NS":     "Ipca Laboratories",
    "JBCHEPHARM.NS":  "J.B. Chemicals & Pharmaceuticals",
    "AJANTPHARM.NS":  "Ajanta Pharma",
    "GRANULES.NS":    "Granules India",
    "NATCOPHARM.NS":  "Natco Pharma",
    "ABBOTINDIA.NS":  "Abbott India",
    "PFIZER.NS":      "Pfizer India",
    "GLAXO.NS":       "GlaxoSmithKline Pharmaceuticals",
    "SANOFI.NS":      "Sanofi India",
    "FORTIS.NS":      "Fortis Healthcare",
    "MAXHEALTH.NS":   "Max Healthcare Institute",
    "NH.NS":          "Narayana Hrudayalaya",
    "MEDANTA.NS":     "Global Health (Medanta)",
    "ASTERDM.NS":     "Aster DM Healthcare",
    "RAINBOW.NS":     "Rainbow Children's Medicare",
    "METROPOLIS.NS":  "Metropolis Healthcare",
    "LAURUSLABS.NS":  "Laurus Labs",
    "DIVISLAB.NS":    "Divi's Laboratories",
    "POLYMED.NS":     "Poly Medicure",

    # ---- More FMCG / Consumer ----
    "EMAMILTD.NS":    "Emami",
    "GILLETTE.NS":    "Gillette India",
    "PGHH.NS":        "Procter & Gamble Hygiene & Health Care",
    "JYOTHYLAB.NS":   "Jyothy Labs",
    "TATACONSUM.NS":  "Tata Consumer Products",
    "VBL.NS":         "Varun Beverages",
    "RADICO.NS":      "Radico Khaitan",
    "USL.NS":         "United Spirits",
    "BIKAJI.NS":      "Bikaji Foods International",
    "ZYDUSWELL.NS":   "Zydus Wellness",
    "BAJAJCON.NS":    "Bajaj Consumer Care",
    "HONAUT.NS":      "Honeywell Automation India",
    "RELAXO.NS":      "Relaxo Footwears",
    "BATAINDIA.NS":   "Bata India",
    "VOLTAS.NS":      "Voltas",
    "BLUESTAR.NS":    "Blue Star",
    "WHIRLPOOL.NS":   "Whirlpool of India",
    "AMBER.NS":       "Amber Enterprises India",

    # ---- More Auto + Auto Ancillaries ----
    "ASHOKLEY.NS":    "Ashok Leyland",
    "BHARATFORG.NS":  "Bharat Forge",
    "BALKRISIND.NS":  "Balkrishna Industries (BKT Tyres)",
    "APOLLOTYRE.NS":  "Apollo Tyres",
    "CEAT.NS":        "CEAT",
    "JKTYRE.NS":      "JK Tyre & Industries",
    "TVSHLTD.NS":     "TVS Holdings",
    "ENDURANCE.NS":   "Endurance Technologies",
    "SUNDARMFIN.NS":  "Sundaram Finance",
    "SUNDRMFAST.NS":  "Sundaram Fasteners",
    "MRPL.NS":        "Mangalore Refinery & Petrochemicals",
    "MAHSCOOTER.NS":  "Maharashtra Scooters",
    "ATULAUTO.NS":    "Atul Auto",
    "GREAVES.NS":     "Greaves Cotton",
    "OLECTRA.NS":     "Olectra Greentech",

    # ---- More Capital Goods / Industrials ----
    "LTIM.NS":        "LTIMindtree",
    "LTOVERSEAS.NS":  "L&T Overseas Investments",
    "CGPOWER.NS":     "CG Power & Industrial Solutions",
    "THERMAX.NS":     "Thermax",
    "KPI.NS":         "KPI Global Infrastructure",
    "KEC.NS":         "KEC International",
    "ASTRAL.NS":      "Astral",
    "FINOLEXIND.NS":  "Finolex Industries",
    "SUPREMEIND.NS":  "Supreme Industries",
    "KIRLOSKAR.NS":   "Kirloskar Brothers",
    "CARBORUNIV.NS":  "Carborundum Universal",
    "GRINDWELL.NS":   "Grindwell Norton",
    "TIMKEN.NS":      "Timken India",
    "SCHAEFFLER.NS":  "Schaeffler India",
    "SKFINDIA.NS":    "SKF India",
    "RAMCOSYS.NS":    "Ramco Systems",
    "GMRP&UI.NS":     "GMR Power & Urban Infra",
    "JKLAKSHMI.NS":   "JK Lakshmi Cement",
    "JKCEMENT.NS":    "JK Cement",
    "RAMCOCEM.NS":    "Ramco Cements",
    "DALBHARAT.NS":   "Dalmia Bharat",
    "ACC.NS":         "ACC Limited",
    "AMBUJACEM.NS":   "Ambuja Cements",
    "SHREECEM.NS":    "Shree Cement",
    "ULTRACEMCO.NS":  "UltraTech Cement",
    "INDIACEM.NS":    "India Cements",
    "BIRLACORPN.NS":  "Birla Corporation",
    "HEIDELBERG.NS":  "Heidelberg Cement India",

    # ---- More Metals & Mining ----
    "WELCORP.NS":     "Welspun Corp",
    "WELSPUN.NS":     "Welspun India",
    "APLAPOLLO.NS":   "APL Apollo Tubes",
    "JINDALSAW.NS":   "Jindal Saw",
    "RATNAMANI.NS":   "Ratnamani Metals & Tubes",
    "MANINFRA.NS":    "Man Infraconstruction",
    "GMDCLTD.NS":     "Gujarat Mineral Development",
    "MOIL.NS":        "MOIL",
    "VEDL.NS":        "Vedanta",
    "HINDZINC.NS":    "Hindustan Zinc",
    "NATIONALUM.NS":  "National Aluminium",
    "JSL.NS":         "Jindal Stainless",
    "NMDC.NS":        "NMDC",
    "KIOCL.NS":       "KIOCL",
    "GRSE.NS":        "Garden Reach Shipbuilders & Engineers",
    "COCHINSHIP.NS":  "Cochin Shipyard",

    # ---- More Power & Energy ----
    "TATAPOWER.NS":   "Tata Power",
    "ADANIPOWER.NS":  "Adani Power",
    "JSWENERGY.NS":   "JSW Energy",
    "TORNTPOWER.NS":  "Torrent Power",
    "CESC.NS":        "CESC",
    "NHPC.NS":        "NHPC",
    "SJVN.NS":        "SJVN",
    "INOXWIND.NS":    "Inox Wind",
    "ORIENTGREEN.NS": "Orient Green Power",
    "RPOWER.NS":      "Reliance Power",
    "IREDA.NS":       "IREDA",
    "ADANIENSOL.NS":  "Adani Energy Solutions",

    # ---- Oil, Gas & Energy ----
    "MGL.NS":         "Mahanagar Gas",
    "GUJGASLTD.NS":   "Gujarat Gas",
    "HPCL.NS":        "Hindustan Petroleum",
    "OIL.NS":         "Oil India",
    "CASTROLIND.NS":  "Castrol India",
    "GULFOIL.NS":     "Gulf Oil Lubricants India",
    "AEGISCHEM.NS":   "Aegis Logistics",

    # ---- Real Estate & Construction ----
    "OBEROIRLTY.NS":  "Oberoi Realty",
    "GODREJPROP.NS":  "Godrej Properties",
    "BRIGADE.NS":     "Brigade Enterprises",
    "PRESTIGE.NS":    "Prestige Estates Projects",
    "SOBHA.NS":       "Sobha",
    "PHOENIXLTD.NS":  "The Phoenix Mills",
    "MAHLIFE.NS":     "Mahindra Lifespace Developers",
    "SUNTECK.NS":     "Sunteck Realty",
    "LODHA.NS":       "Macrotech Developers (Lodha)",
    "IRB.NS":         "IRB Infrastructure Developers",
    "GMRINFRA.NS":    "GMR Airports Infrastructure",

    # ---- Media & Entertainment ----
    "ZEEL.NS":        "Zee Entertainment Enterprises",
    "PVRINOX.NS":     "PVR Inox",
    "SAREGAMA.NS":    "Saregama India",
    "TIPSINDLTD.NS":  "Tips Industries",
    "NETWORK18.NS":   "Network 18 Media & Investments",
    "TV18BRDCST.NS":  "TV18 Broadcast",

    # ---- Chemicals & Specialty ----
    "TATACHEM.NS":    "Tata Chemicals",
    "DEEPAKNTR.NS":   "Deepak Nitrite",
    "AARTIIND.NS":    "Aarti Industries",
    "GUJALKALI.NS":   "Gujarat Alkalies & Chemicals",
    "NAVINFLUOR.NS":  "Navin Fluorine International",
    "SRF.NS":         "SRF",
    "FLUOROCHEM.NS":  "Gujarat Fluorochemicals",
    "ATUL.NS":        "Atul",
    "VINATIORGA.NS":  "Vinati Organics",
    "CLEAN.NS":       "Clean Science & Technology",
    "BASF.NS":        "BASF India",
    "PIDILITIND.NS":  "Pidilite Industries",
    "SOLARINDS.NS":   "Solar Industries India",
    "CHEMPLAST.NS":   "Chemplast Sanmar",
    "EPL.NS":         "EPL",
    "GAEL.NS":        "Gujarat Ambuja Exports",
    "ALKYLAMINE.NS":  "Alkyl Amines Chemicals",

    # ---- New-age & Internet ----
    "MAMAEARTH.NS":   "Honasa Consumer (Mamaearth)",
    "FIVESTAR.NS":    "Five-Star Business Finance",
    "LICI.NS":        "Life Insurance Corporation of India",
    "GOCOLORS.NS":    "Go Fashion (Go Colors)",
    "VEDANTFASH.NS":  "Vedant Fashions",
    "CAMPUS.NS":      "Campus Activewear",
    "METROBRAND.NS":  "Metro Brands",
    "EASEMYTRIP.NS":  "Easy Trip Planners (EaseMyTrip)",
    "CARTRADE.NS":    "CarTrade Tech",
    "FSL.NS":         "Firstsource Solutions",
    "ROUTE.NS":       "Route Mobile",
    "AFFLE.NS":       "Affle (India)",
    "LATENTVIEW.NS":  "LatentView Analytics",
    "HAPPYFORGE.NS":  "Happy Forgings",
    "TBOTEK.NS":      "TBO Tek",
    "AWFIS.NS":       "Awfis Space Solutions",
    "MEDPLUS.NS":     "Medplus Health Services",
    "DELHIVERY.NS":   "Delhivery",
    "PAYTM.NS":       "One 97 Communications (Paytm)",
    "POLICYBZR.NS":   "PB Fintech (Policybazaar)",
    "ZOMATO.NS":      "Zomato (Eternal)",
    "NYKAA.NS":       "FSN E-Commerce Ventures (Nykaa)",

    # ---- Conglomerates & Misc Large-caps ----
    "ITC.NS":         "ITC Limited",
    "GODREJIND.NS":   "Godrej Industries",
    "BIRLAMONEY.NS":  "Aditya Birla Money",
    "ABCAPITAL.NS":   "Aditya Birla Capital",
    "AB.NS":          "Aditya Birla Real Estate",
    "BLISSGVS.NS":    "Bliss GVS Pharma",
    "HONDAPOWER.NS":  "Honda India Power Products",
    "ENGINERSIN.NS":  "Engineers India",
    "RVNL.NS":        "Rail Vikas Nigam",
    "IRCON.NS":       "IRCON International",
    "NBCC.NS":        "NBCC India",
    "HUDCO.NS":       "Housing & Urban Development Corp",
    "BEML.NS":        "BEML",
    "DRREDDY.NS":     "Dr Reddy's Laboratories",
    "INDIANHOTELS.NS":"Indian Hotels (Taj)",
    "EIHOTEL.NS":     "EIH (Oberoi Hotels)",
    "LEMONTREE.NS":   "Lemon Tree Hotels",
    "MAHINDRACIE.NS": "Mahindra CIE Automotive",
    "SUPRAJIT.NS":    "Suprajit Engineering",
    "SUMICHEM.NS":    "Sumitomo Chemical India",
    "DRLALPATH.NS":   "Dr Lal PathLabs",
    "THYROCARE.NS":   "Thyrocare Technologies",
    "BLUEDART.NS":    "Blue Dart Express",
    "MAHLOG.NS":      "Mahindra Logistics",
    "VTL.NS":         "Vardhman Textiles",
    "PAGEIND.NS":     "Page Industries",
    "ARVIND.NS":      "Arvind",
    "RAYMOND.NS":     "Raymond",
    "SIYSIL.NS":      "Siyaram Silk Mills",
}


def search_nse_stocks(query: str, limit: int = 50) -> list[tuple[str, str]]:
    """Return ``(ticker, name)`` tuples whose ticker OR name contains
    ``query`` (case-insensitive). Sorted by relevance — prefix matches first,
    then substring matches, alphabetical within each group.
    """
    q = (query or "").strip().upper()
    if not q:
        return []

    prefix_matches: list[tuple[str, str]] = []
    substr_matches: list[tuple[str, str]] = []
    for ticker, name in NSE_STOCK_NAMES.items():
        plain = ticker.replace(".NS", "")
        name_u = name.upper()
        if plain.startswith(q) or name_u.startswith(q):
            prefix_matches.append((ticker, name))
        elif q in plain or q in name_u:
            substr_matches.append((ticker, name))
    out = (sorted(prefix_matches, key=lambda x: x[1])
           + sorted(substr_matches, key=lambda x: x[1]))
    return out[:limit]


# ---------------------------------------------------------------------------
# Ticker normalisation
# ---------------------------------------------------------------------------
def normalise_ticker(raw: str) -> str:
    """Coerce user-typed input into a yfinance-compatible NSE ticker.

    Rules (mirrors :func:`portfolio_parser._clean_ticker`):
        * Uppercase, strip whitespace.
        * If the symbol already has a ``.`` or starts with ``^`` (index)
          → pass through.
        * Otherwise append ``.NS`` (NSE default for yfinance).
        * Strip common broker segment suffixes (``-EQ``, ``-BE``, ``-BZ``).
    """
    if not raw:
        return ""
    s = str(raw).strip().upper()
    if not s:
        return ""
    if "." in s or s.startswith("^"):
        return s.replace(" ", "")
    # Internal whitespace is virtually never part of a real NSE ticker
    # (the user probably typed "TATA STEEL" meaning TATASTEEL).
    s = s.replace(" ", "")
    for suffix in ("-EQ", "-BE", "-BZ"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
            break
    s = s.rstrip("-_ ").strip()
    if not s:
        return ""
    return f"{s}.NS"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def lazy_load_tickers(
    tickers: Iterable[str],
    start: str,
    end: str,
    regime_method: str = "hmm",
) -> tuple[dict[str, pd.DataFrame], list[str], list[str]]:
    """Fetch + process a list of NSE tickers through the full pipeline.

    Parameters
    ----------
    tickers : iterable of str
        NSE tickers to fetch. Will be deduplicated and normalised.
    start, end : str (YYYY-MM-DD)
        Date range to fetch. Should match the main pipeline's range so
        downstream stats are comparable.
    regime_method : str
        Which regime detector to apply. Should match the main pipeline.

    Returns
    -------
    signaled : dict[ticker, DataFrame]
        Same shape as the main pipeline's output — each ticker mapped
        to a fully-enriched (features + regime + signals + entry/exit
        levels) DataFrame.
    succeeded : list[str]
        Tickers that loaded cleanly.
    failed : list[str]
        Tickers that yfinance couldn't return or that failed the QA gate.
    """
    # Normalise + dedupe.
    norm = sorted({normalise_ticker(t) for t in tickers if normalise_ticker(t)})
    if not norm:
        return {}, [], []

    logger.info("Lazy-loading %d ticker(s): %s", len(norm), norm)

    loader = DataLoader(universe=norm, start=start, end=end)
    data = loader.load_universe()

    # Anything that loaded → goes into the pipeline.
    succeeded = list(data.keys())
    failed = [t for t in norm if t not in data]

    if not data:
        return {}, [], failed

    enriched = FeatureEngineer().compute_universe(data)
    regimed = (RegimeDetector(method=regime_method)
                .fit_transform_universe(enriched))
    signaled = SignalEngine().generate_universe(regimed)
    signaled = {t: add_entry_exit_levels(df) for t, df in signaled.items()}

    # Re-confirm `succeeded` after the universe loop (in case a ticker
    # passed QA but failed feature/signal computation).
    actually_succeeded = list(signaled.keys())
    for t in succeeded:
        if t not in signaled:
            failed.append(t)
    return signaled, actually_succeeded, failed


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Try fetching a couple of tickers outside the default universe.
    result, ok, bad = lazy_load_tickers(
        ["BAJAJ-AUTO.NS", "MOTHERSON.NS", "FAKE-TICKER.NS"],
        start="2020-01-01", end=C.END_DATE,
    )
    print(f"Succeeded ({len(ok)}): {ok}")
    print(f"Failed    ({len(bad)}): {bad}")
    for t, df in result.items():
        print(f"  {t}: {len(df)} bars, last signal = {df['Signal_Strength'].iloc[-1]}")

"""Shared configuration: paths, constants, sample definition, XBRL metric map."""

import base64
import os
import ssl
import sys
import tempfile
from pathlib import Path


def _export_windows_ca_certs() -> None:
    """Export Windows system CA certificates so curl_cffi trusts the corporate proxy.
    """
    if sys.platform != "win32" or os.environ.get("CURL_CA_BUNDLE"):
        return

    pem_path = os.path.join(tempfile.gettempdir(), "python_win_cacerts.pem")

    der_certs: set[bytes] = set()
    for store in ("ROOT", "CA"):
        try:
            for cert, encoding, _trust in ssl.enum_certificates(store):
                if encoding == "x509_asn":
                    der_certs.add(cert)
        except PermissionError:
            pass

    if not der_certs:
        return

    with open(pem_path, "w", encoding="ascii") as f:
        for der in der_certs:
            b64 = base64.b64encode(der).decode("ascii")
            f.write("-----BEGIN CERTIFICATE-----\n")
            for i in range(0, len(b64), 64):
                f.write(b64[i:i + 64] + "\n")
            f.write("-----END CERTIFICATE-----\n")

    os.environ["CURL_CA_BUNDLE"] = pem_path
    os.environ["REQUESTS_CA_BUNDLE"] = pem_path


_export_windows_ca_certs()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# SEC EDGAR API
# ---------------------------------------------------------------------------
SEC_USER_AGENT = "Simon Slansky simon.slansky@outlook.com"
SEC_HEADERS = {"User-Agent": SEC_USER_AGENT}
REQUEST_SLEEP = 0.15

# ---------------------------------------------------------------------------
# Sample selection
# ---------------------------------------------------------------------------
# Number of top companies to process.  Source: the SEC file
# https://www.sec.gov/files/company_tickers.json, whose entries are listed in
# (approximately) descending market-capitalisation order.  The file carries
# no market-cap values and no date; the list used in the thesis was frozen
# to data/universe.csv on UNIVERSE_SNAPSHOT_DATE.  Because the ranking is
# as of that date (not point-in-time), the sample is tilted toward firms
# that are large *today* (look-ahead / survivorship; see Phase-0 decision 1).
# Set to None to use the full SEC EDGAR universe.
SAMPLE_SIZE: int | None = 1000
UNIVERSE_SNAPSHOT_DATE = "see data/universe_meta.json"

# SIC 6000–6999 = Finance, Insurance, Real Estate → incomparable financials
EXCLUDED_SIC_RANGE = range(6000, 7000)


# ---------------------------------------------------------------------------
# Filing validation
# ---------------------------------------------------------------------------
MAX_FILING_LAG_DAYS = 180

# Annual
NUM_YEARS = 17          # XBRL era FY2009-2025; one year consumed by asset-growth lag
MIN_VALID_YEARS = 5
ANNUAL_FLOW_RANGE = (350, 380)   # days

# ---------------------------------------------------------------------------
# Stock-return windows
# ---------------------------------------------------------------------------
# Window starts this many TRADING days after the first trading day on or
# after the filing date (see scripts/volatility.py).
POST_FILING_LAG_TRADING_DAYS = 2
ANNUAL_RETURN_WINDOW = 365   # calendar days (window length after start)
MIN_TRADING_DAYS_ANNUAL = 200   # minimum closes -> min returns = 199

# Price source: Yahoo Finance via the ``yfinance`` package.
#   data/prices/{T}.csv      adjusted closes (auto_adjust=True: split- AND
#                            dividend-adjusted) -> used for returns/volatility.
#   data/prices_raw/{T}.csv  split-adjusted close + split events
#                            (auto_adjust=False) -> used to rebuild the
#                            as-traded price for market capitalisation.
# Price download range (cached for reproducibility).
PRICE_START = "2009-01-01"
PRICE_END: str | None = None    # None = today at first download
MARKET_TICKER = "SPY"           # market proxy for market / idiosyncratic vol
# Data cut-off: a post-filing window whose calendar end lies after the last
# trading day of the market series is set to NaN (never truncated).
MIN_FIRM_COVERAGE = 0.90        # legacy; the post-filing coverage filter is OFF

# Pre-filing liquidity rule (replaces the post-filing 90% filter, #5):
# a filing is "liquid" if the stock has >= PRE_LIQ_MIN_DAYS closes in the
# PRE_LIQ_WINDOW trading days before the filing date.
PRE_LIQ_WINDOW = 252
PRE_LIQ_MIN_DAYS = 200

# Pre-filing volatility windows (trading days ending the day before filing).
PRE_VOL_SHORT = 60
PRE_VOL_LONG = 252

# ---------------------------------------------------------------------------
# Winsorization
# ---------------------------------------------------------------------------
WINSORIZE_LOWER = 0.01
WINSORIZE_UPPER = 0.99

# ---------------------------------------------------------------------------
# XBRL tag → internal metric mapping  (tag, metric_name, priority)
# Lower priority number = preferred source.
# ---------------------------------------------------------------------------
METRICS_WITH_PRIORITY: list[tuple[str, str, int]] = [
    # Total Assets
    ("Assets",                          "total_assets",          1),
    # Asset / liability sub-components (for ratio computation)
    ("AssetsCurrent",                   "assets_current",        1),
    # Total Liabilities
    ("Liabilities",                     "total_liabilities",     1),
    ("LiabilitiesCurrent",              "liabilities_current",   1),
    # Stockholders' Equity (for fallback liabilities computation)
    # Priority: tag including NCI first, so the accounting identity A = L + E holds
    ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                                        "stockholders_equity",   1),
    ("StockholdersEquity",              "stockholders_equity",   2),
    # Net Income (flow).  CONSOLIDATED basis (incl. NCI), consistent with
    # equity incl. NCI above, so that ROA and leverage share one reporting
    # entity (#13).  ProfitLoss is only reported when NCI exists; otherwise
    # NetIncomeLoss is by definition the consolidated figure.
    ("ProfitLoss",                      "net_income",            1),
    ("NetIncomeLoss",                   "net_income",            2),
    ("IncomeLossFromContinuingOperations",
                                        "net_income",            3),
    # Operating Income (flow) — EBIT proxy for operating ROA
    ("OperatingIncomeLoss",             "operating_income",      1),
    # Operating Cash Flow (flow) — for OCF-to-assets ratio (robustness)
    ("NetCashProvidedByUsedInOperatingActivities",
                                        "operating_cash_flow",   1),
    ("NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
                                        "operating_cash_flow",   2),
    # ── Diagnostic single-tag metrics (never locked; used for the tag-overlap
    #    diagnostic #14/#15 and for book-to-market / market cap) ──────────
    ("NetIncomeLoss",                   "ni_parent",             1),
    ("ProfitLoss",                      "ni_consol",             1),
    ("NetIncomeLossAttributableToNoncontrollingInterest",
                                        "ni_nci",                1),
    ("NetCashProvidedByUsedInOperatingActivities",
                                        "ocf_total",             1),
    ("NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
                                        "ocf_cont",              1),
    ("StockholdersEquity",              "equity_parent",         1),
]

FLOW_METRICS = {"net_income", "operating_income", "operating_cash_flow",
                "ni_parent", "ni_consol", "ni_nci", "ocf_total", "ocf_cont"}

# Share counts (unit "shares"), matched to the 10-K by accession number.
# dei cover-page count (date close to filing) preferred; balance-sheet count
# as fallback.
SHARE_TAGS: list[tuple[str, str]] = [
    ("dei", "EntityCommonStockSharesOutstanding"),
    ("us-gaap", "CommonStockSharesOutstanding"),
]

# Materiality threshold for paired-tag disagreements (#15): share of assets.
TAG_DIFF_MATERIAL = 0.01

# ---------------------------------------------------------------------------
# Tags that are economically equivalent and safe to mix within a firm.
# Operating cash flow: the "ContinuingOperations" variant was introduced in
#   later XBRL taxonomies; for firms without material discontinued operations
#   (the vast majority) both tags report the identical figure.
# Net income: ProfitLoss (consolidated, incl. NCI) is preferred; NetIncomeLoss
#   is used only when ProfitLoss is absent, which under US GAAP (ASC 810)
#   means there is no NCI, so the two are the same consolidated quantity.
#   Disagreements in firm-years reporting both (= NCI) are quantified in
#   data/tag_overlap.csv.
# IncomeLossFromContinuingOperations is NOT included — it deliberately
#   excludes discontinued operations, which can be material.
# ---------------------------------------------------------------------------
EQUIVALENT_TAG_GROUPS: dict[str, set[str]] = {
    "operating_cash_flow": {
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    },
    "net_income": {
        "NetIncomeLoss",
        "ProfitLoss",
    },
}

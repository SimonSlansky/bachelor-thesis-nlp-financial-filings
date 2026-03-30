"""Shared configuration: paths, constants, sample definition, XBRL metric map."""

from pathlib import Path

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
# Number of top companies (by SEC market-cap rank) to process.
# Set to None to use the full SEC EDGAR universe.
SAMPLE_SIZE: int | None = 1000

# SIC 6000–6999 = Finance, Insurance, Real Estate → incomparable financials
EXCLUDED_SIC_RANGE = range(6000, 7000)


# ---------------------------------------------------------------------------
# Filing validation
# ---------------------------------------------------------------------------
MAX_FILING_LAG_DAYS = 180

# Annual
NUM_YEARS = 16          # extra year consumed by asset-growth lag
MIN_VALID_YEARS = 5
ANNUAL_FLOW_RANGE = (350, 380)   # days

# Quarterly (Q1-Q3 only; Q4 is implicit in 10-K)
NUM_QUARTERS = 45
MIN_VALID_QUARTERS = 4
QUARTERLY_FLOW_RANGE = (65, 120)  # days

# ---------------------------------------------------------------------------
# Stock-return windows
# ---------------------------------------------------------------------------
POST_FILING_LAG_DAYS = 2
ANNUAL_RETURN_WINDOW = 365   # calendar days
QUARTERLY_RETURN_WINDOW = 63
MIN_TRADING_DAYS_ANNUAL = 200
MIN_TRADING_DAYS_QUARTERLY = 40
MIN_FIRM_COVERAGE = 0.90

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
    ("StockholdersEquity",              "stockholders_equity",   1),
    ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                                        "stockholders_equity",   2),
    # Net Income (flow)
    ("NetIncomeLoss",                   "net_income",            1),
    ("ProfitLoss",                      "net_income",            2),
    ("IncomeLossFromContinuingOperations",
                                        "net_income",            3),
    # Operating Income (flow) — EBIT proxy for operating ROA
    ("OperatingIncomeLoss",             "operating_income",      1),
    # Operating Cash Flow (flow) — for accruals ratio
    ("NetCashProvidedByUsedInOperatingActivities",
                                        "operating_cash_flow",   1),
    # EPS (flow) — for descriptive statistics
    ("EarningsPerShareDiluted",         "eps_diluted",           1),
    ("EarningsPerShareBasic",           "eps_diluted",           2),
]

FLOW_METRICS = {"net_income", "operating_income", "operating_cash_flow", "eps_diluted"}
INSTANT_METRICS = {m for _, m, _ in METRICS_WITH_PRIORITY if m not in FLOW_METRICS}

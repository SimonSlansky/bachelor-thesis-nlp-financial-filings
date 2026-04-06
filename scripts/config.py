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
NUM_YEARS = 17          # XBRL era FY2009-2025; one year consumed by asset-growth lag
MIN_VALID_YEARS = 5
ANNUAL_FLOW_RANGE = (350, 380)   # days

# ---------------------------------------------------------------------------
# Stock-return windows
# ---------------------------------------------------------------------------
POST_FILING_LAG_DAYS = 2
ANNUAL_RETURN_WINDOW = 365   # calendar days
MIN_TRADING_DAYS_ANNUAL = 200
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
    # Priority: tag including NCI first, so the accounting identity A = L + E holds
    ("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                                        "stockholders_equity",   1),
    ("StockholdersEquity",              "stockholders_equity",   2),
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
    ("NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
                                        "operating_cash_flow",   2),
]

FLOW_METRICS = {"net_income", "operating_income", "operating_cash_flow"}

# ---------------------------------------------------------------------------
# Tags that are economically equivalent and safe to mix within a firm.
# Operating cash flow: the "ContinuingOperations" variant was introduced in
#   later XBRL taxonomies; for firms without material discontinued operations
#   (the vast majority) both tags report the identical figure.
# Net income: ProfitLoss includes NCI, NetIncomeLoss excludes it.  For large
#   US firms NCI is typically <0.5 % of total assets, making the accruals and
#   ROA impact negligible (Hribar & Collins 2002 treat OCF the same way).
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

"""Shared post-filing volatility / return machinery.

Single source of truth for window alignment, used by ``returns.py`` (365-day
horizon) and ``build_volatility_horizons.py`` (short horizons).

Window definition (matches tex/chapters/04_methodology.tex, eq. vol):

* Day 0 is the first trading day on or after the EDGAR filing date.
* The window starts ``POST_FILING_LAG_TRADING_DAYS`` (= 2) *trading* days
  after day 0 and spans ``h`` calendar days from that start.
* Daily log returns are computed only from closes inside the window, so the
  filing-reaction close-to-close moves are excluded.
* Volatility is the sample SD of daily log returns x sqrt(252).  A window
  with fewer than ``min_returns`` returns, or a non-positive SD, is NaN
  (never clipped).

Lagged volatility (control ln sigma_{i,t}) uses the same firm's window
anchored on the filing of fiscal year t-1 *only* (no multi-year gaps), and is
truncated at the last trading day before the current filing so it can never
overlap the outcome window.

Prices are cached per ticker in ``data/prices/`` so reruns are reproducible
and do not depend on later yfinance revisions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA_DIR, POST_FILING_LAG_TRADING_DAYS, PRICE_START, PRICE_END

ANNUALISATION_FACTOR = float(np.sqrt(252))
PRICE_DIR = DATA_DIR / "prices"


# ── price loading with on-disk cache ─────────────────────────────────────────

def load_closes(ticker: str, refresh: bool = False) -> pd.Series | None:
    """Return adjusted daily closes for *ticker* (cached CSV)."""
    PRICE_DIR.mkdir(exist_ok=True)
    path = PRICE_DIR / f"{ticker}.csv"
    if path.exists() and not refresh:
        s = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        s = s.dropna()
        return s if len(s) else None

    import yfinance as yf
    prices = yf.download(ticker, start=PRICE_START, end=PRICE_END,
                         progress=False, auto_adjust=True)
    if prices is None or prices.empty:
        return None
    if isinstance(prices.columns, pd.MultiIndex):
        prices = prices.droplevel("Ticker", axis=1)
    if "Close" not in prices.columns:
        return None
    s = prices["Close"].dropna()
    s.index = pd.DatetimeIndex(s.index).tz_localize(None)
    s.name = "Close"
    s.to_csv(path)
    return s if len(s) else None


# ── window statistics ────────────────────────────────────────────────────────

def window_stats(closes: pd.Series, filing_date: pd.Timestamp,
                 window_days: int, min_returns: int,
                 end_cap: pd.Timestamp | None = None) -> tuple[float, float]:
    """(log return, annualised vol) over the post-filing window.

    *end_cap* (inclusive) truncates the window; used for lagged windows so
    that they end strictly before the next filing.
    """
    idx = closes.index
    day0 = idx.searchsorted(pd.Timestamp(filing_date), side="left")
    start_pos = day0 + POST_FILING_LAG_TRADING_DAYS
    if start_pos >= len(idx):
        return np.nan, np.nan
    start = idx[start_pos]
    end = start + pd.Timedelta(days=window_days)
    if end_cap is not None:
        end = min(end, pd.Timestamp(end_cap))
    w = closes.loc[start:end]
    if len(w) < 2:
        return np.nan, np.nan
    daily = np.log(w / w.shift(1)).dropna()
    if len(daily) < min_returns:
        return np.nan, np.nan
    vol = float(daily.std(ddof=1) * ANNUALISATION_FACTOR)
    if not np.isfinite(vol) or vol <= 0:
        return np.nan, np.nan
    log_ret = float(np.log(w.iloc[-1] / w.iloc[0]))
    return log_ret, vol


PRE_FILING_TRADING_DAYS = 60
PRE_FILING_MIN_RETURNS = 40


def pre_filing_vol(closes: pd.Series, filing_date: pd.Timestamp,
                   n_days: int = PRE_FILING_TRADING_DAYS,
                   min_returns: int = PRE_FILING_MIN_RETURNS) -> float:
    """Annualised vol of the *n_days* daily returns that END on the last
    trading day strictly before the filing date.

    Everything used is public before the filing, so it is a valid control
    for the market's risk state at the time the 10-K is released.
    """
    idx = closes.index
    day0 = idx.searchsorted(pd.Timestamp(filing_date), side="left")
    # closes idx[day0-n_days-1 .. day0-1]  ->  n_days returns
    lo = max(day0 - n_days - 1, 0)
    w = closes.iloc[lo:day0]
    if len(w) < 2:
        return np.nan
    daily = np.log(w / w.shift(1)).dropna()
    if len(daily) < min_returns:
        return np.nan
    vol = float(daily.std(ddof=1) * ANNUALISATION_FACTOR)
    return vol if np.isfinite(vol) and vol > 0 else np.nan


def current_and_lagged(closes: pd.Series, firm: pd.DataFrame,
                       window_days: int, min_returns: int
                       ) -> list[dict]:
    """Compute outcome and lagged stats for every filing of one firm.

    *firm* must contain ``fiscal_year`` and ``filing_date``.  Returns one dict
    per row with keys ``ret``, ``vol``, ``lag_vol`` (index-aligned to *firm*).
    """
    f = firm.sort_values("fiscal_year")
    fds = pd.to_datetime(f["filing_date"], errors="coerce")
    fys = f["fiscal_year"].astype(int).to_numpy()
    by_fy = dict(zip(fys, fds))
    out = []
    for ix, fy, fd in zip(f.index, fys, fds):
        rec = {"_ix": ix, "ret": np.nan, "vol": np.nan, "lag_vol": np.nan}
        if pd.isna(fd):
            out.append(rec)
            continue
        rec["ret"], rec["vol"] = window_stats(closes, fd, window_days,
                                              min_returns)
        prev_fd = by_fy.get(fy - 1)          # strictly t-1, no gaps
        if prev_fd is not None and pd.notna(prev_fd) and prev_fd < fd:
            cap = fd - pd.Timedelta(days=1)   # end before current filing
            _, rec["lag_vol"] = window_stats(closes, prev_fd, window_days,
                                             min_returns, end_cap=cap)
        out.append(rec)
    return out

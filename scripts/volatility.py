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

from config import (DATA_DIR, POST_FILING_LAG_TRADING_DAYS, PRICE_START,
                    PRICE_END, MARKET_TICKER, PRE_LIQ_WINDOW,
                    PRE_VOL_SHORT, PRE_VOL_LONG)

ANNUALISATION_FACTOR = float(np.sqrt(252))
PRICE_DIR = DATA_DIR / "prices"
PRICE_RAW_DIR = DATA_DIR / "prices_raw"

_MARKET: pd.Series | None = None


def market_closes() -> pd.Series:
    """Adjusted closes of the market proxy (cached)."""
    global _MARKET
    if _MARKET is None:
        m = load_closes(MARKET_TICKER)
        if m is None:
            raise RuntimeError(f"no price data for market proxy {MARKET_TICKER}")
        _MARKET = m
    return _MARKET


def data_cutoff() -> pd.Timestamp:
    """Last trading day in the price data (market proxy).  Windows ending
    after this date are incomplete and are set to NaN (#36)."""
    return market_closes().index.max()


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


# ── window statistics ─────────────────────────────────────────────────────

def window_bounds(closes: pd.Series, filing_date: pd.Timestamp,
                  window_days: int, end_cap: pd.Timestamp | None = None
                  ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    """(start, end) calendar bounds of the post-filing window, or None.

    Day 0 is located on the MARKET calendar so that a firm with a missing
    close on the filing day does not shift its window."""
    mkt_idx = market_closes().index
    day0 = mkt_idx.searchsorted(pd.Timestamp(filing_date), side="left")
    start_pos = day0 + POST_FILING_LAG_TRADING_DAYS
    if start_pos >= len(mkt_idx):
        return None
    start = mkt_idx[start_pos]
    end = start + pd.Timedelta(days=window_days)
    if end_cap is not None:
        end = min(end, pd.Timestamp(end_cap))
    elif end > data_cutoff():
        return None
    return start, end


def _window_returns(s: pd.Series, start, end) -> pd.Series:
    w = s.loc[start:end]
    return np.log(w / w.shift(1)).dropna()


def window_market_stats(closes: pd.Series, filing_date: pd.Timestamp,
                        window_days: int, min_returns: int
                        ) -> tuple[float, float]:
    """(market vol, idiosyncratic vol) over the same post-filing window.

    Idiosyncratic vol is the annualised SD of residuals from a within-window
    market-model regression r_i = a + b r_m + e (#30)."""
    bounds = window_bounds(closes, filing_date, window_days)
    if bounds is None:
        return np.nan, np.nan
    rm = _window_returns(market_closes(), *bounds)
    ri = _window_returns(closes, *bounds)
    mkt_vol = (float(rm.std(ddof=1) * ANNUALISATION_FACTOR)
               if len(rm) >= min_returns else np.nan)
    j = pd.concat([ri, rm], axis=1, join="inner").dropna()
    if len(j) < max(min_returns, 3):
        return mkt_vol, np.nan
    x, y = j.iloc[:, 1].to_numpy(), j.iloc[:, 0].to_numpy()
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    dof = len(y) - 2
    idio = float(np.sqrt((resid ** 2).sum() / dof) * ANNUALISATION_FACTOR) if dof > 0 else np.nan
    return mkt_vol, (idio if np.isfinite(idio) and idio > 0 else np.nan)


def earnings_in_window(ea_dates: np.ndarray, closes: pd.Series,
                       filing_date: pd.Timestamp, window_days: int) -> float:
    """1 if an earnings release (8-K Item 2.02) falls inside the window."""
    bounds = window_bounds(closes, filing_date, window_days)
    if bounds is None:
        return np.nan
    if ea_dates is None or len(ea_dates) == 0:
        return 0.0
    s, e = np.datetime64(bounds[0]), np.datetime64(bounds[1])
    return float(((ea_dates >= s) & (ea_dates <= e)).any())


def vol_excluding_ea(closes: pd.Series, filing_date: pd.Timestamp,
                     window_days: int, min_returns: int,
                     ea_dates: np.ndarray | None) -> float:
    """Outcome-window volatility without the returns of earnings-release
    reaction days: the first trading day on or after each 8-K Item 2.02 date
    and the next trading day (covers after-close releases)."""
    bounds = window_bounds(closes, filing_date, window_days)
    if bounds is None:
        return np.nan
    r = _window_returns(closes, *bounds)
    if ea_dates is not None and len(ea_dates):
        cal = closes.index
        drop = set()
        for d in ea_dates:
            k = cal.searchsorted(pd.Timestamp(d), side="left")
            drop.update(cal[k:k + 2])
        r = r[~r.index.isin(list(drop))]
    if len(r) < min_returns:
        return np.nan
    v = float(r.std(ddof=1) * ANNUALISATION_FACTOR)
    return v if np.isfinite(v) and v > 0 else np.nan


def pre_filing_stats(closes: pd.Series, filing_date: pd.Timestamp) -> dict:
    """Pre-filing (public-at-filing) price variables.

    All windows END on the last market trading day strictly before the
    filing date:
      pre_vol      — annualised vol, last PRE_VOL_SHORT trading days
      pre_vol_252  — annualised vol, last PRE_VOL_LONG trading days
      ret_pre_12m  — log return over the last 252 trading days
      n_pre_days   — firm closes in the last PRE_LIQ_WINDOW market days
                     (pre-filing liquidity rule, #5)
    """
    mkt_idx = market_closes().index
    day0 = mkt_idx.searchsorted(pd.Timestamp(filing_date), side="left")
    out = {"pre_vol": np.nan, "pre_vol_252": np.nan,
           "ret_pre_12m": np.nan, "n_pre_days": 0}
    if day0 < 2:
        return out
    last = mkt_idx[day0 - 1]
    first_liq = mkt_idx[max(day0 - PRE_LIQ_WINDOW, 0)]
    out["n_pre_days"] = int(closes.loc[first_liq:last].notna().sum())
    for n, key, min_ret in ((PRE_VOL_SHORT, "pre_vol", int(PRE_VOL_SHORT * 2 / 3)),
                            (PRE_VOL_LONG, "pre_vol_252", int(PRE_VOL_LONG * 2 / 3))):
        first = mkt_idx[max(day0 - n - 1, 0)]
        r = _window_returns(closes, first, last)
        if len(r) >= min_ret:
            v = float(r.std(ddof=1) * ANNUALISATION_FACTOR)
            out[key] = v if np.isfinite(v) and v > 0 else np.nan
    w = closes.loc[mkt_idx[max(day0 - 253, 0)]:last]
    if len(w) >= int(PRE_LIQ_WINDOW * 2 / 3):
        out["ret_pre_12m"] = float(np.log(w.iloc[-1] / w.iloc[0]))
    return out


def as_traded_close(ticker: str, date: pd.Timestamp) -> float:
    """As-traded (unadjusted) close on the last trading day before *date*.

    data/prices_raw/{T}.csv holds the split-adjusted close and split events
    (fetch_market_data.py); multiplying by the product of all split ratios
    AFTER the date undoes the split adjustment.  Dividends do not affect
    yfinance's (non-auto-adjusted) Close."""
    p = PRICE_RAW_DIR / f"{ticker}.csv"
    if not p.exists():
        return np.nan
    raw = pd.read_csv(p, index_col=0, parse_dates=True)
    raw.index = pd.DatetimeIndex(raw.index).tz_localize(None)
    before = raw.loc[raw.index < pd.Timestamp(date)]
    if before.empty:
        return np.nan
    d = before.index[-1]
    close = float(before["Close"].iloc[-1])
    splits = raw.loc[raw.index > d, "Stock Splits"]
    factor = float(splits[splits > 0].prod()) if (splits > 0).any() else 1.0
    return close * factor

def window_stats(closes: pd.Series, filing_date: pd.Timestamp,
                 window_days: int, min_returns: int,
                 end_cap: pd.Timestamp | None = None) -> tuple[float, float]:
    """(log return, annualised vol) over the post-filing window.

    *end_cap* (inclusive) truncates the window; used for lagged windows so
    that they end strictly before the next filing.  An *uncapped* window
    whose nominal end lies past the data cut-off is NaN (incomplete).
    """
    bounds = window_bounds(closes, filing_date, window_days, end_cap)
    if bounds is None:
        return np.nan, np.nan
    start, end = bounds
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

"""Compute post-filing log stock returns and realised volatility via yfinance."""

import numpy as np
import pandas as pd
import yfinance as yf

from config import POST_FILING_LAG_DAYS

ANNUALISATION_FACTOR = np.sqrt(252)


def compute_returns_and_volatility(
    df: pd.DataFrame,
    date_col: str,
    return_col: str,
    vol_col: str,
    window_days: int,
    min_trading_days: int,
) -> pd.DataFrame:
    """Add *return_col* and *vol_col* to *df* from yfinance prices.

    Parameters
    ----------
    df : Panel with columns ``ticker``, ``filing_date``, and *date_col*.
    date_col : Column used for the merge key (``year_end`` or ``quarter_end``).
    return_col : Name of the log-return column to create.
    vol_col : Name of the annualised-volatility column to create.
    window_days : Calendar-day window after filing.
    min_trading_days : Minimum trading days required for valid output.

    Returns
    -------
    A copy of *df* with *return_col* and *vol_col* merged in.
    """
    if df.empty:
        df[return_col] = np.nan
        df[vol_col] = np.nan
        return df

    min_date = pd.to_datetime(df[date_col].min())
    max_filing = pd.to_datetime(df["filing_date"]).max()
    download_end = max_filing + pd.Timedelta(days=POST_FILING_LAG_DAYS + window_days + 10)

    results: list[dict] = []

    for ticker in sorted(df["ticker"].unique()):
        prices = yf.download(
            ticker,
            start=min_date,
            end=download_end,
            progress=False,
            auto_adjust=True,
        )
        if prices.empty:
            print(f"  {ticker}: no price data")
            continue

        # yfinance ≥1.2 returns MultiIndex columns even for a single ticker;
        # flatten to simple column names so .iloc[] returns scalars.
        if isinstance(prices.columns, pd.MultiIndex):
            prices = prices.droplevel("Ticker", axis=1)

        if "Close" not in prices.columns:
            print(f"  {ticker}: no Close column")
            continue

        closes = prices[["Close"]].dropna()
        firm = df.loc[df["ticker"] == ticker]
        usable = 0

        for _, row in firm.iterrows():
            fd = pd.to_datetime(row["filing_date"], errors="coerce")
            if pd.isna(fd):
                continue
            start = fd + pd.Timedelta(days=POST_FILING_LAG_DAYS)
            end = start + pd.Timedelta(days=window_days)
            w = closes.loc[start:end]
            if len(w) < min_trading_days:
                continue

            log_ret = float(np.log(w.iloc[-1]["Close"] / w.iloc[0]["Close"]))
            daily_log_rets = np.log(w["Close"] / w["Close"].shift(1)).dropna()
            ann_vol = float(daily_log_rets.std() * ANNUALISATION_FACTOR)

            results.append({
                "ticker": ticker,
                date_col: row[date_col],
                return_col: log_ret,
                vol_col: ann_vol,
            })
            usable += 1

        n = len(firm)
        print(f"  {ticker}: {usable}/{n} periods")

    df_ret = pd.DataFrame(results)

    # Coverage filtering is deferred to panel.drop_low_return_coverage(),
    # which runs AFTER drop_earliest_year and cap_fiscal_year shrink the
    # denominator.  Pre-filtering here would zero returns for firms whose
    # post-trim coverage actually exceeds the threshold.

    if df_ret.empty:
        df = df.copy()
        df[return_col] = np.nan
        df[vol_col] = np.nan
    else:
        df = df.merge(df_ret, on=["ticker", date_col], how="left")

    return df

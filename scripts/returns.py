"""Compute post-filing log stock returns, realised volatility and the lagged
(previous-fiscal-year) volatility control.  Window alignment lives in
``volatility.py``."""

import numpy as np
import pandas as pd

from volatility import load_closes, current_and_lagged


def compute_returns_and_volatility(
    df: pd.DataFrame,
    date_col: str,
    return_col: str,
    vol_col: str,
    window_days: int,
    min_trading_days: int,
    lag_col: str = "lagged_vol",
) -> pd.DataFrame:
    """Add *return_col*, *vol_col* and *lag_col* to *df*.

    ``lag_col`` is the volatility of the same-length window after the
    fiscal-year t-1 filing, truncated before the current filing date (NaN if
    year t-1 is absent).  *date_col* is kept for API compatibility only.
    """
    df = df.copy()
    for c in (return_col, vol_col, lag_col):
        df[c] = np.nan
    if df.empty:
        return df

    min_returns = max(min_trading_days - 1, 2)
    for ticker in sorted(df["ticker"].unique()):
        closes = load_closes(ticker)
        firm = df.loc[df["ticker"] == ticker]
        if closes is None:
            print(f"  {ticker}: no price data")
            continue
        recs = current_and_lagged(closes, firm, window_days, min_returns)
        for r in recs:
            df.loc[r["_ix"], [return_col, vol_col, lag_col]] = (
                r["ret"], r["vol"], r["lag_vol"])
        usable = sum(np.isfinite(r["vol"]) for r in recs)
        print(f"  {ticker}: {usable}/{len(firm)} periods")

    # Coverage filtering is deferred to panel.drop_low_return_coverage().
    return df

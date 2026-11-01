"""Compute short post-filing volatility windows (5d … 180d) and their lags.

Reads ``data/annual_panel.csv`` for the (ticker, fiscal_year, filing_date)
tuples, uses the cached prices from ``data/prices/`` (see volatility.py) and
writes ``data/volatility_horizons.csv`` with, for each horizon h,

* ``vol_{h}d``        — outcome window (starts 2 trading days after filing),
* ``lagged_vol_{h}d`` — same window after the fiscal-year t-1 filing,
                        truncated before the current filing; NaN if t-1 is
                        missing (no multi-year lags).

Usage:
    python scripts/build_volatility_horizons.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR, PRE_LIQ_MIN_DAYS
from volatility import (load_closes, current_and_lagged, pre_filing_stats,
                        window_market_stats, earnings_in_window,
                        as_traded_close, data_cutoff)

# (window_days, column, minimum number of daily RETURNS)
# The 5-day window holds only 3–4 closes after the 2-trading-day skip, so it
# admits 3 returns; zero / degenerate SDs are set to NaN, never clipped.
# vol_365d uses the same machinery as the short horizons (incl. the data
# cut-off rule); it supersedes returns.py's vol_next_year in regressions.
HORIZONS: list[tuple[int, str, int]] = [
    (5,   "vol_5d",   3),
    (10,  "vol_10d",  5),
    (30,  "vol_30d",  14),
    (90,  "vol_90d",  44),
    (180, "vol_180d", 89),
    (365, "vol_365d", 199),
]

OUT_PATH = DATA_DIR / "volatility_horizons.csv"


def _load_ea() -> dict[str, np.ndarray]:
    p = DATA_DIR / "earnings_dates.csv"
    if not p.exists():
        print("  [warn] earnings_dates.csv missing: run fetch_earnings_dates.py;"
              " earnings dummies will be NaN")
        return {}
    ea = pd.read_csv(p, parse_dates=["ea_date"]).dropna(subset=["ea_date"])
    return {t: g["ea_date"].values.astype("datetime64[ns]")
            for t, g in ea.groupby("ticker")}


def main() -> None:
    panel_path = DATA_DIR / "annual_panel.csv"
    if not panel_path.exists():
        print(f"Missing {panel_path}; run build_annual_panel.py first.",
              file=sys.stderr)
        sys.exit(1)

    panel = pd.read_csv(panel_path, parse_dates=["filing_date", "year_end"])
    panel = panel.dropna(subset=["filing_date"])
    print(f"Loaded {len(panel):,} filings, {panel['ticker'].nunique()} tickers")
    print(f"Data cut-off (last market trading day): {data_cutoff().date()}")
    ea_map = _load_ea()

    # First-year filings dropped from the panel (asset-growth lag year).
    # They are used only as the t-1 anchor for lagged volatility, exactly
    # as returns.py does for the 365-day horizon.
    lag_path = DATA_DIR / "lag_filings.csv"
    lag_rows = (pd.read_csv(lag_path, parse_dates=["filing_date"])
                if lag_path.exists() else
                pd.DataFrame(columns=["ticker", "fiscal_year", "filing_date"]))
    if not lag_path.exists():
        print("  [warn] lag_filings.csv missing: first-year lags will be NaN")

    frames: list[pd.DataFrame] = []
    tickers = sorted(panel["ticker"].unique())
    for i, ticker in enumerate(tickers, 1):
        cols = ["ticker", "year_end", "fiscal_year", "filing_date"]
        firm = panel.loc[panel["ticker"] == ticker, cols]
        out = firm[["ticker", "year_end"]].copy()
        extra = lag_rows.loc[(lag_rows["ticker"] == ticker)
                             & ~lag_rows["fiscal_year"].isin(firm["fiscal_year"])]
        extra = extra.drop_duplicates("fiscal_year")
        extra.index = [f"_lag{j}" for j in range(len(extra))]
        firm_all = pd.concat([firm, extra])
        closes = load_closes(ticker)
        if closes is None:
            print(f"  [{i:>3d}/{len(tickers)}] {ticker:<6s}  no price data")
            continue
        fds = firm["filing_date"].tolist()
        for window_days, col, min_ret in HORIZONS:
            recs = current_and_lagged(closes, firm_all, window_days, min_ret)
            m = {r["_ix"]: r for r in recs}
            out[col] = [m[ix]["vol"] for ix in out.index]
            out[f"lagged_{col}"] = [m[ix]["lag_vol"] for ix in out.index]
            h = col.split("_")[1]
            ms = [window_market_stats(closes, fd, window_days, min_ret) for fd in fds]
            out[f"mkt_vol_{h}"] = [a for a, _ in ms]
            out[f"idio_vol_{h}"] = [b for _, b in ms]
            ea = ea_map.get(ticker) if ea_map else None
            out[f"ea_in_{h}"] = ([earnings_in_window(ea, closes, fd, window_days)
                                  for fd in fds] if ea_map else np.nan)
        pre = pd.DataFrame([pre_filing_stats(closes, fd) for fd in fds],
                           index=out.index)
        out = out.join(pre)
        out["liquid_pre"] = out["n_pre_days"] >= PRE_LIQ_MIN_DAYS
        out["filing_month"] = [fd.month for fd in fds]
        out["price_pre"] = [as_traded_close(ticker, fd) for fd in fds]
        frames.append(out)
        if i % 50 == 0:
            print(f"  [{i}/{len(tickers)}]", flush=True)

    out = pd.concat(frames, ignore_index=True)

    # Market capitalisation and book-to-market (public at filing).
    fin = panel[["ticker", "year_end", "shares_out", "stockholders_equity",
                 "equity_parent"]] if "shares_out" in panel else None
    if fin is not None:
        out = out.merge(fin, on=["ticker", "year_end"], how="left")
        out["mcap"] = out["price_pre"] * out["shares_out"]
        out.loc[~(out["mcap"] > 0), "mcap"] = np.nan
        out["log_mcap"] = np.log(out["mcap"])
        # Book equity attributable to common holders (parent); fall back to
        # total equity when the parent tag is absent.
        be = out["equity_parent"].fillna(out["stockholders_equity"])
        out["bm"] = be / out["mcap"]
        out = out.drop(columns=["shares_out", "stockholders_equity",
                                "equity_parent"])
    else:
        print("  [warn] annual_panel.csv has no shares_out — rerun build_annual_panel.py")

    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {OUT_PATH.name}: {len(out):,} rows")
    for _, col, _ in HORIZONS:
        h = col.split("_")[1]
        print(f"  {col:10s} cov={out[col].notna().mean():6.1%}  "
              f"lag={out['lagged_' + col].notna().mean():6.1%}  "
              f"mkt={out[f'mkt_vol_{h}'].notna().mean():6.1%}  "
              f"idio={out[f'idio_vol_{h}'].notna().mean():6.1%}  "
              f"EA-share={pd.to_numeric(out[f'ea_in_{h}'], errors='coerce').mean():.2f}")
    for c in ("pre_vol", "pre_vol_252", "ret_pre_12m", "log_mcap", "bm"):
        if c in out:
            print(f"  {c:12s} cov={out[c].notna().mean():6.1%}  median={out[c].median():.4f}")
    print(f"  liquid_pre   share={out['liquid_pre'].mean():6.1%}")


if __name__ == "__main__":
    main()

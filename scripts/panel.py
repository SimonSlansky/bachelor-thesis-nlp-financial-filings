"""Panel construction utilities shared by annual and quarterly pipelines."""

import numpy as np
import pandas as pd

from config import WINSORIZE_LOWER, WINSORIZE_UPPER


# ── missing-metric imputation ─────────────────────────────────────────────

def compute_missing_components(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing total_liabilities from total_assets − stockholders_equity."""
    df = df.copy()

    # total_liabilities = total_assets − stockholders_equity
    if {"total_assets", "stockholders_equity"}.issubset(df.columns):
        mask = df["total_liabilities"].isna() & df["total_assets"].notna() & df["stockholders_equity"].notna()
        df.loc[mask, "total_liabilities"] = df.loc[mask, "total_assets"] - df.loc[mask, "stockholders_equity"]

    return df


def impute_balance_sheet(df: pd.DataFrame, date_col: str, limit: int = 1) -> pd.DataFrame:
    """Forward- then backward-fill balance-sheet metrics within each firm."""
    df = df.sort_values(["ticker", date_col]).reset_index(drop=True)
    for col in ("total_assets", "total_liabilities"):
        if col in df.columns:
            before = df[col].isna().sum()
            df[col] = df.groupby("ticker")[col].ffill(limit=limit)
            df[col] = df.groupby("ticker")[col].bfill(limit=limit)
            filled = before - df[col].isna().sum()
            if filled:
                print(f"  {col}: imputed {filled} values (±{limit} period)")
    return df


# ── derived financial ratios ──────────────────────────────────────────────

def add_financial_ratios(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Compute financial ratios from raw metrics.

    Ratios: log_total_assets, leverage, roa, asset_growth,
            current_ratio, accruals, ocf_to_assets, operating_roa.
    Winsorizes ratio variables at the configured percentiles.
    """
    df = df.sort_values(["ticker", date_col]).reset_index(drop=True)

    df["log_total_assets"] = np.where(df["total_assets"] > 0, np.log(df["total_assets"]), np.nan)

    df["leverage"] = np.where(df["total_assets"] > 0, df["total_liabilities"] / df["total_assets"], np.nan)

    df["roa"] = np.where(df["total_assets"] > 0, df["net_income"] / df["total_assets"], np.nan)

    # Asset growth: YoY for both annual (shift 1) and quarterly (shift 4)
    is_quarterly = "quarter_end" in df.columns
    lag_periods = 4 if is_quarterly else 1
    lag = df.groupby("ticker")["total_assets"].shift(lag_periods)
    df["asset_growth"] = np.where(lag > 0, (df["total_assets"] - lag) / lag, np.nan)

    df["current_ratio"] = np.where(
        df["liabilities_current"] > 0,
        df["assets_current"] / df["liabilities_current"],
        np.nan,
    )

    if "operating_cash_flow" in df.columns:
        df["accruals"] = np.where(
            df["total_assets"] > 0,
            (df["net_income"] - df["operating_cash_flow"]) / df["total_assets"],
            np.nan,
        )
        df["ocf_to_assets"] = np.where(
            df["total_assets"] > 0,
            df["operating_cash_flow"] / df["total_assets"],
            np.nan,
        )

    if "operating_income" in df.columns:
        df["operating_roa"] = np.where(
            df["total_assets"] > 0,
            df["operating_income"] / df["total_assets"],
            np.nan,
        )

    # winsorize
    for col in ("leverage", "roa", "asset_growth", "current_ratio",
                "accruals", "ocf_to_assets", "operating_roa"):
        if col not in df.columns:
            continue
        s = df[col].dropna()
        if len(s) > 0:
            lo, hi = s.quantile(WINSORIZE_LOWER), s.quantile(WINSORIZE_UPPER)
            df[col] = df[col].clip(lower=lo, upper=hi)

    return df


# ── transition-period & duplicate cleanup (annual only) ───────────────────

def filter_transitions_and_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate filing dates and non-annual spacing rows."""
    df = df.sort_values(["ticker", "year_end"]).reset_index(drop=True)
    n_before = len(df)

    df["_days"] = pd.to_datetime(df["year_end"]).groupby(df["ticker"]).diff().dt.days
    df["_dev"] = (df["_days"] - 365).abs()
    df["_has_ni"] = df["net_income"].notna().astype(int)

    # resolve duplicate filing dates: keep row with net_income + closest to 365-day spacing
    df["_fd_str"] = df["filing_date"].astype(str)
    df["_dup"] = df.duplicated(subset=["ticker", "_fd_str"], keep=False)

    if df["_dup"].any():
        df["_score"] = df["_has_ni"] * 10000 - df["_dev"]
        keep_idx = df.loc[df["_dup"]].groupby(["ticker", "_fd_str"])["_score"].idxmax()
        drop_mask = df["_dup"] & ~df.index.isin(keep_idx)
        df = df.loc[~drop_mask]

    # remove non-annual spacing (< 300 or > 400 days between consecutive year-ends)
    # recompute after dedup since rows may have been dropped
    df["_days"] = pd.to_datetime(df["year_end"]).groupby(df["ticker"]).diff().dt.days
    is_transition = ((df["_days"] < 300) | (df["_days"] > 400)).fillna(False)
    df = df.loc[~is_transition]

    df = df.drop(columns=[c for c in df.columns if c.startswith("_")])
    removed = n_before - len(df)
    if removed:
        print(f"  Removed {removed} transition/duplicate rows ({n_before} → {len(df)})")
    return df.reset_index(drop=True)


def drop_earliest_year(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the earliest year per firm (consumed by asset-growth lag)."""
    earliest = df.groupby("ticker")["year_end"].transform("min")
    mask = df["year_end"] != earliest
    dropped = (~mask).sum()
    print(f"  Dropped {dropped} earliest-year rows used for asset-growth lag")
    return df.loc[mask].reset_index(drop=True)


# ── lagged volatility ─────────────────────────────────────────────────────

def add_lagged_volatility(df: pd.DataFrame, vol_col: str, date_col: str) -> pd.DataFrame:
    """Create ``lagged_vol`` as previous-period realised volatility per firm."""
    df = df.sort_values(["ticker", date_col]).reset_index(drop=True)
    df["lagged_vol"] = df.groupby("ticker")[vol_col].shift(1)
    return df


# ── output helpers ────────────────────────────────────────────────────────

ANNUAL_COLS = [
    "ticker", "company_name", "sic", "fiscal_year_end",
    "year_end", "filing_date", "fiscal_year", "accession_number",
    "net_income", "total_assets", "total_liabilities",
    "operating_income", "operating_cash_flow", "eps_diluted",
    "return_next_year", "vol_next_year", "lagged_vol",
    "log_total_assets", "leverage", "roa", "asset_growth",
    "current_ratio", "accruals", "ocf_to_assets", "operating_roa",
]

QUARTERLY_COLS = [
    "ticker", "company_name", "sic", "fiscal_year_end",
    "quarter_end", "filing_date",
    "fiscal_year", "fiscal_quarter", "accession_number",
    "net_income", "total_assets", "total_liabilities",
    "operating_income", "operating_cash_flow", "eps_diluted",
    "return_next_q", "vol_next_q", "lagged_vol",
    "log_total_assets", "leverage", "roa", "asset_growth",
    "current_ratio", "accruals", "ocf_to_assets", "operating_roa",
]


def save_panel(df: pd.DataFrame, path, columns: list[str]) -> None:
    """Select existing columns from *columns* and save to CSV."""
    cols = [c for c in columns if c in df.columns]
    out = df[cols]
    out.to_csv(path, index=False)
    date_col = "year_end" if "year_end" in cols else "quarter_end"
    print(f"  Saved {path.name}: {out['ticker'].nunique()} firms × {len(out)} obs "
          f"[{out[date_col].min()} — {out[date_col].max()}]")

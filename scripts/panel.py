"""Panel construction utilities for the annual (10-K) pipeline."""

import numpy as np
import pandas as pd

from config import (
    WINSORIZE_LOWER, WINSORIZE_UPPER, METRICS_WITH_PRIORITY,
    EQUIVALENT_TAG_GROUPS,
)


# ── per-firm tag locking ──────────────────────────────────────────────────

def _metrics_with_alternatives() -> dict[str, list[str]]:
    """Return {metric_name: [tag1, tag2, ...]} for metrics with >1 source tag."""
    from collections import defaultdict
    tags_per_metric: dict[str, list[str]] = defaultdict(list)
    for tag, metric, _pri in METRICS_WITH_PRIORITY:
        tags_per_metric[metric].append(tag)
    return {m: tags for m, tags in tags_per_metric.items() if len(tags) > 1}


def lock_firm_tags(df: pd.DataFrame) -> pd.DataFrame:
    """For each (firm, metric) with multiple source tags, keep only the
    tag the firm uses most frequently.  This ensures within-firm time-series
    consistency and prevents mixing economically different XBRL concepts.

    Tags listed in ``EQUIVALENT_TAG_GROUPS`` are treated as interchangeable
    (e.g. both OCF tags are taxonomy variants of the same concept) and are
    exempt from locking.

    Requires ``_tag_{metric}`` columns produced by ``extract_annual_facts``.
    Returns *df* with inconsistent values set to NaN and a summary printed.
    """
    df = df.copy()
    multi = _metrics_with_alternatives()
    summary_rows: list[dict] = []

    for metric, tags in multi.items():
        tag_col = f"_tag_{metric}"
        if tag_col not in df.columns:
            continue

        equiv = EQUIVALENT_TAG_GROUPS.get(metric, set())

        # For each firm, find the tag used in the most years
        firm_tag_counts = (
            df.loc[df[tag_col].notna()]
            .groupby(["ticker", tag_col])
            .size()
            .reset_index(name="n")
        )
        if firm_tag_counts.empty:
            continue
        dominant = (
            firm_tag_counts
            .sort_values("n", ascending=False)
            .drop_duplicates(subset="ticker", keep="first")
            .set_index("ticker")[tag_col]
        )

        # Null out values where the observation's tag ≠ the firm's locked tag,
        # UNLESS both the observation's tag and the locked tag belong to the
        # same equivalence group.
        before_valid = df[metric].notna().sum()
        for ticker, locked_tag in dominant.items():
            mask = (
                (df["ticker"] == ticker)
                & df[tag_col].notna()
                & (df[tag_col] != locked_tag)
            )
            if equiv:
                # Keep rows where BOTH tags are in the equivalence group
                obs_tags = df.loc[mask, tag_col]
                safe = obs_tags.isin(equiv) & (locked_tag in equiv)
                mask = mask & ~safe
            df.loc[mask, metric] = np.nan
            df.loc[mask, tag_col] = np.nan
        after_valid = df[metric].notna().sum()
        nulled = before_valid - after_valid

        # Summary statistics per tag
        for tag in tags:
            n_firms = (dominant == tag).sum()
            summary_rows.append({
                "metric": metric, "tag": tag,
                "firms": n_firms,
                "pct": n_firms / len(dominant) * 100 if len(dominant) else 0,
            })

        if equiv:
            print(f"  {metric}: equivalent tags accepted, nulled {nulled} non-equivalent values")
        elif nulled:
            print(f"  {metric}: locked per-firm tags, nulled {nulled} cross-tag values")
        else:
            print(f"  {metric}: all values already consistent")

    # Save diagnostic summary
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print("\n  Tag distribution across firms:")
        for _, row in summary_df.iterrows():
            print(f"    {row['metric']:25s} {row['tag']:60s} -> {row['firms']:4.0f} firms ({row['pct']:.1f}%)")

    return df


def save_tag_diagnostics(df: pd.DataFrame, path) -> None:
    """Save per-firm tag choices to a CSV for the thesis appendix."""
    multi = _metrics_with_alternatives()
    rows: list[dict] = []
    for metric, _tags in multi.items():
        tag_col = f"_tag_{metric}"
        if tag_col not in df.columns:
            continue
        firm_tags = (
            df.loc[df[tag_col].notna()]
            .groupby("ticker")[tag_col]
            .agg(lambda s: s.mode().iloc[0] if len(s) > 0 else np.nan)
        )
        for ticker, tag in firm_tags.items():
            rows.append({"ticker": ticker, "metric": metric, "locked_tag": tag})
    if rows:
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"  Tag diagnostics -> {path.name}")


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
            current_ratio, ocf_to_assets.
    Winsorization is NOT done here — call ``winsorize_ratios`` on the
    final sample after all filtering.
    """
    df = df.sort_values(["ticker", date_col]).reset_index(drop=True)

    df["log_total_assets"] = np.where(df["total_assets"] > 0, np.log(df["total_assets"]), np.nan)

    df["leverage"] = np.where(df["total_assets"] > 0, df["total_liabilities"] / df["total_assets"], np.nan)

    df["roa"] = np.where(df["total_assets"] > 0, df["net_income"] / df["total_assets"], np.nan)

    # Asset growth: YoY (shift 1 period = 1 fiscal year)
    # Guard: only compute when consecutive rows are ~1 year apart;
    # FYE changes can leave multi-year gaps after transition filtering.
    lag = df.groupby("ticker")["total_assets"].shift(1)
    days_gap = pd.to_datetime(df[date_col]).groupby(df["ticker"]).diff().dt.days
    valid_gap = (days_gap >= 300) & (days_gap <= 400)
    df["asset_growth"] = np.where((lag > 0) & valid_gap, (df["total_assets"] - lag) / lag, np.nan)

    df["current_ratio"] = np.where(
        df["liabilities_current"] > 0,
        df["assets_current"] / df["liabilities_current"],
        np.nan,
    )

    if "operating_cash_flow" in df.columns:
        df["ocf_to_assets"] = np.where(
            df["total_assets"] > 0,
            df["operating_cash_flow"] / df["total_assets"],
            np.nan,
        )

    return df


WINSORIZE_COLS = ("leverage", "roa", "asset_growth", "current_ratio",
                  "ocf_to_assets")


def winsorize_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """Clip ratio columns at configured percentiles on the FINAL sample."""
    df = df.copy()
    for col in WINSORIZE_COLS:
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
        print(f"  Removed {removed} transition/duplicate rows ({n_before} -> {len(df)})")
    return df.reset_index(drop=True)


def drop_earliest_year(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the earliest year per firm (consumed by asset-growth lag)."""
    earliest = df.groupby("ticker")["year_end"].transform("min")
    mask = df["year_end"] != earliest
    dropped = (~mask).sum()
    print(f"  Dropped {dropped} earliest-year rows used for asset-growth lag")
    return df.loc[mask].reset_index(drop=True)


# ── firm-level return coverage filter ──────────────────────────────────────

def drop_low_return_coverage(
    df: pd.DataFrame, vol_col: str, min_coverage: float,
) -> pd.DataFrame:
    """Drop firms where the share of non-NaN *vol_col* obs < *min_coverage*."""
    stats = df.groupby("ticker")[vol_col].agg(["count", "size"])
    stats["cov"] = stats["count"] / stats["size"]
    keep = stats.loc[stats["cov"] >= min_coverage].index
    n_before = df["ticker"].nunique()
    df = df.loc[df["ticker"].isin(keep)].reset_index(drop=True)
    n_after = df["ticker"].nunique()
    dropped = n_before - n_after
    if dropped:
        print(f"  Dropped {dropped} firms with <{min_coverage:.0%} return coverage "
              f"({n_before} -> {n_after} firms)")
    return df


# ── fiscal-year cap ────────────────────────────────────────────────────────

def cap_fiscal_year(df: pd.DataFrame, max_fy: int) -> pd.DataFrame:
    """Remove observations with fiscal_year > *max_fy*."""
    mask = df["fiscal_year"] <= max_fy
    removed = (~mask).sum()
    if removed:
        print(f"  Removed {removed} obs with fiscal_year > {max_fy}")
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
    "year_end", "filing_date", "fiscal_year", "accession_number", "form_type",
    "net_income", "total_assets", "total_liabilities",
    "operating_income", "operating_cash_flow",
    "return_next_year", "vol_next_year", "lagged_vol",
    "log_total_assets", "leverage", "roa", "asset_growth",
    "current_ratio", "ocf_to_assets",
]


def save_panel(df: pd.DataFrame, path, columns: list[str]) -> None:
    """Select existing columns from *columns* and save to CSV."""
    cols = [c for c in columns if c in df.columns]
    out = df[cols]
    out.to_csv(path, index=False)
    print(f"  Saved {path.name}: {out['ticker'].nunique()} firms x {len(out)} obs "
          f"[{out['year_end'].min()} - {out['year_end'].max()}]")

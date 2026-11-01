"""Panel construction utilities for the annual (10-K) pipeline."""

import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    WINSORIZE_LOWER, WINSORIZE_UPPER, METRICS_WITH_PRIORITY,
    EQUIVALENT_TAG_GROUPS,
)


# ── per-firm tag locking ──────────────────────────────────────────────────

# Filled by lock_firm_tags(); written by save_tag_diagnostics() (#16).
LOCK_SUMMARY: list[dict] = []

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
        n_switchers = int(
            firm_tag_counts.groupby("ticker")[tag_col].nunique().gt(1).sum())
        LOCK_SUMMARY.append({
            "metric": metric, "values_nulled": int(nulled),
            "values_before": int(before_valid),
            "firms_with_multiple_tags": n_switchers,
            "firms": int(len(dominant)),
        })

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
    if LOCK_SUMMARY:
        p = Path(path).with_name("tag_locking_summary.csv")
        pd.DataFrame(LOCK_SUMMARY).to_csv(p, index=False)
        print(f"  Tag-locking losses -> {p.name}")


def tag_overlap_diagnostics(df: pd.DataFrame, path) -> pd.DataFrame:
    """Paired-tag disagreement in firm-years that report both tags (#14/#15).

    Differences are scaled by total assets (i.e. in ROA / OCF-to-assets
    units).  'Material' = |diff| / assets > TAG_DIFF_MATERIAL.
    """
    from config import TAG_DIFF_MATERIAL
    pairs = [("net_income", "ni_consol", "ni_parent"),
             ("operating_cash_flow", "ocf_total", "ocf_cont")]
    rows = []
    for metric, a, b in pairs:
        if not {a, b, "total_assets"}.issubset(df.columns):
            continue
        m = df[a].notna() & df[b].notna() & (df["total_assets"] > 0)
        d = ((df.loc[m, a] - df.loc[m, b]).abs() / df.loc[m, "total_assets"])
        if d.empty:
            continue
        rows.append({
            "metric": metric, "tag_a": a, "tag_b": b, "n_both": int(m.sum()),
            "median_pp": 100 * d.median(), "mean_pp": 100 * d.mean(),
            "p90_pp": 100 * d.quantile(.90), "p95_pp": 100 * d.quantile(.95),
            "p99_pp": 100 * d.quantile(.99), "max_pp": 100 * d.max(),
            "share_identical": float((d < 1e-9).mean()),
            "n_material": int((d > TAG_DIFF_MATERIAL).sum()),
            "share_material": float((d > TAG_DIFF_MATERIAL).mean()),
            "material_threshold": TAG_DIFF_MATERIAL,
        })
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    print(f"  Tag-overlap diagnostic -> {Path(path).name}")
    for _, r in out.iterrows():
        print(f"    {r.metric:20s} n={r.n_both:5d}  median={r.median_pp:.3f}pp "
              f"p99={r.p99_pp:.3f}pp  material={r.n_material}")
    return out


# ── missing-metric imputation ─────────────────────────────────────────────

def compute_missing_components(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing total_liabilities from total_assets − stockholders_equity."""
    df = df.copy()
    df["imputed_liab_identity"] = False

    # total_liabilities = total_assets − stockholders_equity
    if {"total_assets", "stockholders_equity"}.issubset(df.columns):
        mask = df["total_liabilities"].isna() & df["total_assets"].notna() & df["stockholders_equity"].notna()
        df.loc[mask, "total_liabilities"] = df.loc[mask, "total_assets"] - df.loc[mask, "stockholders_equity"]
        df.loc[mask, "imputed_liab_identity"] = True
        print(f"  total_liabilities: {int(mask.sum())} values from A - E identity")

    return df


def impute_balance_sheet(df: pd.DataFrame, date_col: str, limit: int = 1) -> pd.DataFrame:
    """Forward-fill balance-sheet metrics within each firm (past values only).

    Backward filling was removed: it copied the year t+1 balance sheet, which
    is published ~one year after the year-t filing, into year-t controls
    (look-ahead).  Only an immediately preceding fiscal year is used.
    """
    df = df.sort_values(["ticker", date_col]).reset_index(drop=True)
    consecutive = df.groupby("ticker")["fiscal_year"].diff() == 1
    for col in ("total_assets", "total_liabilities"):
        if col in df.columns:
            flag = f"imputed_ffill_{col}"
            df[flag] = False
            before = df[col].isna().sum()
            prev = df.groupby("ticker")[col].shift(1)
            fill = df[col].isna() & consecutive & prev.notna()
            df.loc[fill, col] = prev[fill]
            df.loc[fill, flag] = True
            filled = before - df[col].isna().sum()
            if filled:
                print(f"  {col}: forward-filled {filled} values from year t-1")
    return df


def consecutive_lag(df: pd.DataFrame, col: str, by: str = "ticker",
                    time: str = "fiscal_year", diff: bool = False) -> pd.Series:
    """Within-firm lag (or first difference) of *col* that is NaN unless the
    previous row is exactly fiscal year t-1.  Index-aligned to *df*."""
    order = df.sort_values([by, time])
    prev = order.groupby(by)[col].shift(1)
    ok = order.groupby(by)[time].diff() == 1
    lag = prev.where(ok)
    res = order[col] - lag if diff else lag
    return res.reindex(df.index)


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
                  "ocf_to_assets", "bm", "ret_pre_12m")


def winsorize_ratios(df: pd.DataFrame, cols=WINSORIZE_COLS) -> pd.DataFrame:
    """Clip ratio columns at the 1st/99th percentiles of *df*.

    Must be called on the REGRESSION sample (after all dropna filters), as
    stated in the methodology; the stored panels are left unwinsorised.
    """
    df = df.copy()
    for col in cols:
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
    """Deprecated: ``lagged_vol`` is now produced by returns.py from prices
    (t-1 window truncated before the current filing).  Kept as a fallback
    that at least enforces strictly consecutive fiscal years."""
    if "lagged_vol" in df.columns:
        return df
    df = df.sort_values(["ticker", date_col]).reset_index(drop=True)
    df["lagged_vol"] = consecutive_lag(df, vol_col)
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
    # equity / shares for market cap & book-to-market (built downstream)
    "stockholders_equity", "equity_parent", "shares_out", "shares_source",
    # imputation flags (#17)
    "imputed_liab_identity", "imputed_ffill_total_assets",
    "imputed_ffill_total_liabilities",
    # sample bookkeeping
    "n_years_firm", "sic_snapshot",
]


def save_panel(df: pd.DataFrame, path, columns: list[str]) -> None:
    """Select existing columns from *columns* and save to CSV."""
    cols = [c for c in columns if c in df.columns]
    out = df[cols]
    out.to_csv(path, index=False)
    print(f"  Saved {path.name}: {out['ticker'].nunique()} firms x {len(out)} obs "
          f"[{out['year_end'].min()} - {out['year_end'].max()}]")

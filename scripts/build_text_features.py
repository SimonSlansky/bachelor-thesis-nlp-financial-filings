"""End-to-end driver for the textual-feature pipeline.

Reads:
  * ``data/text_sections.csv``          — Item 1A / Item 7 narratives per filing.
  * ``data/annual_panel.csv``           — financial / volatility panel.
  * ``data/lm_master.csv``              — Loughran–McDonald master dictionary.
  * ``data/volatility_horizons.csv``    — (optional) short-horizon vols.

Writes:
  * ``data/annual_panel_text.csv`` — the panel extended with sentiment_mda,
    risk_1a, words_mda, words_1a, textsim, delta_sentiment, delta_risk,
    delta_roa, divergence, divergence_sign, and (if present) the
    short-horizon volatilities ``vol_30d``, ``vol_90d``, ``vol_180d``
    together with their within-firm one-period lags.
  * ``data/lambda.json``           — the calibrated λ used for divergence.
"""

from __future__ import annotations

import sys

import pandas as pd

from config import DATA_DIR
from lm_loader import load_lm_dictionary
from text_features import (
    add_delta_roa,
    add_divergence,
    calibrate_lambda,
    compute_section_scores,
    compute_textsim,
    load_text_sections,
    merge_text_features,
    save_lambda_diagnostic,
)

HORIZON_COLS = ["vol_5d", "vol_10d", "vol_30d", "vol_90d", "vol_180d", "vol_365d"]


def _merge_filing_metadata(p: pd.DataFrame, text_df: pd.DataFrame) -> pd.DataFrame:
    """Historical SIC at filing (#72).  The section source (main document vs
    exhibit) is not available: measuring it would need the 10-Ks again (#26)."""
    meta_p = DATA_DIR / "filing_metadata.csv"
    if meta_p.exists():
        m = pd.read_csv(meta_p).drop_duplicates(["ticker", "fiscal_year"], keep="last")
        keep = [c for c in ("sic_at_filing", "sic_source") if c in m.columns]
        p = p.drop(columns=[c for c in keep + ["src_1a", "src_7"] if c in p])
        p = p.merge(m[["ticker", "fiscal_year", *keep]], on=["ticker", "fiscal_year"],
                    how="left")
        p["sic_hist"] = p["sic_at_filing"].fillna(p["sic"])
        print(f"  filing metadata: SIC coverage {p['sic_at_filing'].notna().mean():.1%}, "
              f"SIC changed vs today in {(p['sic_at_filing'].notna() & (p['sic_at_filing'] != p['sic'])).mean():.1%}")
    else:
        print("  [warn] filing_metadata.csv missing: run backfill_filing_metadata.py")
        p["sic_hist"] = p["sic"]
    return p


def _coverage_by_year(p: pd.DataFrame) -> None:
    """Actual extraction coverage on the full panel, by fiscal year (#23)."""
    g = p.groupby("fiscal_year")
    cov = pd.DataFrame({
        "filings": g.size(),
        "item_1a_ok": g["words_1a"].apply(lambda s: (s >= 100).sum()),
        "item_7_ok": g["words_mda"].apply(lambda s: (s >= 100).sum()),
    })
    cov["item_1a_rate"] = cov["item_1a_ok"] / cov["filings"]
    cov["item_7_rate"] = cov["item_7_ok"] / cov["filings"]
    tot = cov[["filings", "item_1a_ok", "item_7_ok"]].sum()
    cov.loc["all"] = [*tot, tot["item_1a_ok"] / tot["filings"], tot["item_7_ok"] / tot["filings"]]
    cov.to_csv(DATA_DIR / "text_coverage_by_year.csv")
    print(f"  coverage (all years): Item 1A {cov.loc['all', 'item_1a_rate']:.1%}, "
          f"Item 7 {cov.loc['all', 'item_7_rate']:.1%} -> text_coverage_by_year.csv")


def main() -> None:
    text_path = DATA_DIR / "text_sections.csv"
    panel_path = DATA_DIR / "annual_panel.csv"
    out_path = DATA_DIR / "annual_panel_text.csv"

    if not text_path.exists():
        print(f"Missing {text_path}; run build_annual_panel.py first.",
              file=sys.stderr)
        sys.exit(1)
    if not panel_path.exists():
        print(f"Missing {panel_path}; run build_annual_panel.py first.",
              file=sys.stderr)
        sys.exit(1)

    print("Loading inputs …")
    text_df = load_text_sections()
    panel = pd.read_csv(panel_path)
    lm = load_lm_dictionary()
    print(f"  text rows: {len(text_df):,}  panel rows: {len(panel):,}  "
          f"LM words: pos={len(lm['positive'])}, neg={len(lm['negative'])}, "
          f"unc={len(lm['uncertainty'])}, lit={len(lm['litigious'])}")

    print("\nScoring sentiment + risk rates …")
    scores = compute_section_scores(text_df, lm)
    n_sent = scores["sentiment_mda"].notna().sum()
    n_risk = scores["risk_1a"].notna().sum()
    n_unc  = scores["unc_1a"].notna().sum()
    n_lit  = scores["lit_1a"].notna().sum()
    print(f"  sentiment_mda: {n_sent:,} / {len(scores):,} non-null")
    print(f"  risk_1a:       {n_risk:,} / {len(scores):,} non-null")
    print(f"  unc_1a:        {n_unc:,} / {len(scores):,} non-null")
    print(f"  lit_1a:        {n_lit:,} / {len(scores):,} non-null")

    print("\nComputing TF-IDF YoY similarity …")
    sims = compute_textsim(text_df)
    print(f"  textsim:       {sims['textsim'].notna().sum():,} non-null "
          f"(first-year obs are NaN)")

    print("\nMerging into panel + first differences …")
    p = merge_text_features(panel, scores, sims)
    p = add_delta_roa(p)

    # Length features used by H1 (verbosity of risk-factor disclosure) and as a
    # control / robustness for H2 (density of risk language). log(1+x) handles
    # the rare zero/missing-section cases gracefully.
    import numpy as np  # local import keeps the module-level imports tidy
    p["log_words_1a"] = np.log(p["words_1a"].clip(lower=1))
    p["log_words_mda"] = np.log(p["words_mda"].clip(lower=1))
    # Ratio-free count variables (#33): log(1 + dictionary-word counts).
    p["log_n_unc_1a"] = np.log1p(p["n_unc_1a"])
    p["log_n_lit_1a"] = np.log1p(p["n_lit_1a"])

    p = _merge_filing_metadata(p, text_df)
    _coverage_by_year(p)

    print("Calibrating λ …")
    lam = calibrate_lambda(p, "delta_sentiment", "delta_roa")
    n_lam = p[["delta_sentiment", "delta_roa"]].dropna().shape[0]
    save_lambda_diagnostic(lam, n_lam)

    p = add_divergence(p, lam)

    horizons_path = DATA_DIR / "volatility_horizons.csv"
    if horizons_path.exists():
        print("\nMerging short-horizon volatilities …")
        h = pd.read_csv(horizons_path, parse_dates=["year_end"])
        # The text panel keeps year_end as a string (it survives a CSV
        # round-trip); convert both sides to datetime for a clean merge.
        p["year_end"] = pd.to_datetime(p["year_end"], errors="coerce")
        hcols = [c for c in HORIZON_COLS if c in h.columns]
        lcols = [f"lagged_{c}" for c in hcols if f"lagged_{c}" in h.columns]
        if len(lcols) != len(hcols):
            sys.exit("volatility_horizons.csv lacks lagged_* columns; "
                     "re-run build_volatility_horizons.py")
        extra = [c for c in h.columns
                 if c not in ("ticker", "year_end", *hcols, *lcols)]
        p = p.drop(columns=[c for c in extra if c in p.columns])
        p = p.merge(h[["ticker", "year_end", *hcols, *lcols, *extra]],
                    on=["ticker", "year_end"], how="left")
        # Lagged short-horizon vols come from build_volatility_horizons.py:
        # window after the fiscal-year t-1 filing, truncated before the
        # current filing (no overlap, no multi-year gaps).
        p = p.sort_values(["ticker", "year_end"]).reset_index(drop=True)
        for c in HORIZON_COLS:
            if c in p.columns:
                print(f"    {c:10s} coverage = {p[c].notna().mean():6.1%}")
    else:
        print(f"\n[skip] {horizons_path.name} not found — "
              "run build_volatility_horizons.py to enable the"
              " post-filing-decay robustness check.")

    p.to_csv(out_path, index=False)
    print(f"\nSaved {out_path.name}: {p['ticker'].nunique()} firms x {len(p):,} obs")
    cov = p[["delta_sentiment", "delta_risk", "textsim", "divergence"]].notna().mean()
    print("  coverage:")
    for k, v in cov.items():
        print(f"    {k:18s} {v:6.1%}")


if __name__ == "__main__":
    main()

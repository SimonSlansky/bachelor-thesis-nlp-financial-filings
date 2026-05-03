"""End-to-end driver for the textual-feature pipeline.

Reads:
  * ``data/text_sections.csv``   — Item 1A / Item 7 narratives per filing.
  * ``data/annual_panel.csv``    — financial / volatility panel.
  * ``data/lm_master.csv``       — Loughran–McDonald master dictionary.

Writes:
  * ``data/annual_panel_text.csv`` — the panel extended with sentiment_mda,
    risk_1a, words_mda, words_1a, textsim, delta_sentiment, delta_risk,
    delta_roa, divergence, divergence_sign.
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
    merge_text_features,
    save_lambda_diagnostic,
)


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
    text_df = pd.read_csv(text_path)
    panel = pd.read_csv(panel_path)
    lm = load_lm_dictionary()
    print(f"  text rows: {len(text_df):,}  panel rows: {len(panel):,}  "
          f"LM words: pos={len(lm['positive'])}, neg={len(lm['negative'])}, "
          f"unc={len(lm['uncertainty'])}, lit={len(lm['litigious'])}")

    print("\nScoring sentiment + risk rates …")
    scores = compute_section_scores(text_df, lm)
    n_sent = scores["sentiment_mda"].notna().sum()
    n_risk = scores["risk_1a"].notna().sum()
    print(f"  sentiment_mda: {n_sent:,} / {len(scores):,} non-null")
    print(f"  risk_1a:       {n_risk:,} / {len(scores):,} non-null")

    print("\nComputing TF-IDF YoY similarity …")
    sims = compute_textsim(text_df)
    print(f"  textsim:       {sims['textsim'].notna().sum():,} non-null "
          f"(first-year obs are NaN)")

    print("\nMerging into panel + first differences …")
    p = merge_text_features(panel, scores, sims)
    p = add_delta_roa(p)

    print("Calibrating λ …")
    lam = calibrate_lambda(p, "delta_sentiment", "delta_roa")
    n_lam = p[["delta_sentiment", "delta_roa"]].dropna().shape[0]
    save_lambda_diagnostic(lam, n_lam)

    p = add_divergence(p, lam)

    p.to_csv(out_path, index=False)
    print(f"\nSaved {out_path.name}: {p['ticker'].nunique()} firms x {len(p):,} obs")
    cov = p[["delta_sentiment", "delta_risk", "textsim", "divergence"]].notna().mean()
    print("  coverage:")
    for k, v in cov.items():
        print(f"    {k:18s} {v:6.1%}")


if __name__ == "__main__":
    main()

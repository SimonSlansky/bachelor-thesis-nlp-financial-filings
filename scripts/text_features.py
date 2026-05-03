"""Text features for the volatility–disclosure panel.

Implements the textual variables defined in
``tex/chapters/04_methodology.tex`` (§sec:variables, §ssec:divergence):

* ``sentiment_mda``  — net tone of Item 7 (MD&A): (pos − neg) / N_words.
* ``risk_1a``        — share of risk words in Item 1A (Risk Factors):
                       (uncertainty + litigious) / N_words.
* ``delta_sentiment``, ``delta_risk`` — within-firm year-on-year first
                                        differences.
* ``textsim``        — cosine similarity of TF–IDF vectors built from
                       (item_1a + item_7) versus the firm's prior-year
                       filing.
* ``divergence``     — Δsentiment − λ · Δroa, where λ is calibrated by a
                       within-firm OLS of Δsentiment on Δroa.

The module is deliberately small: pure pandas / numpy / scikit-learn,
no caching, no parallelism.  See ``scripts/build_text_features.py`` for
the end-to-end driver.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR
from lm_loader import load_lm_dictionary

# A word is a lowercase run of letters of length >= 2.  This matches the
# tokenisation implicit in the Loughran-McDonald dictionary (which has
# no numerals, hyphens, or apostrophes in its sentiment lists).
_TOKEN_RE = re.compile(r"[a-z]+")


# ── Tokenisation & per-document scoring ───────────────────────────────────

def tokenise(text: str) -> list[str]:
    """Return a list of lowercase alphabetic tokens of length >= 2."""
    if not isinstance(text, str) or not text:
        return []
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) > 1]


def _score_tokens(tokens: list[str], lm: dict[str, set[str]]) -> dict[str, float]:
    """Compute net-tone and risk-rate from a token list.

    Returns a dict with raw word counts and the two per-document rates.
    A document with fewer than 100 tokens returns NaN rates (treated as
    a parsing failure for that section).
    """
    n = len(tokens)
    if n < 100:
        return {
            "n_words": n,
            "n_pos": 0, "n_neg": 0, "n_unc": 0, "n_lit": 0,
            "sentiment": np.nan, "risk": np.nan,
        }
    counts = Counter(tokens)
    n_pos = sum(c for w, c in counts.items() if w in lm["positive"])
    n_neg = sum(c for w, c in counts.items() if w in lm["negative"])
    n_unc = sum(c for w, c in counts.items() if w in lm["uncertainty"])
    n_lit = sum(c for w, c in counts.items() if w in lm["litigious"])
    return {
        "n_words": n,
        "n_pos": n_pos, "n_neg": n_neg, "n_unc": n_unc, "n_lit": n_lit,
        "sentiment": (n_pos - n_neg) / n,
        "risk": (n_unc + n_lit) / n,
    }


def compute_section_scores(text_df: pd.DataFrame,
                           lm: dict[str, set[str]] | None = None,
                           verbose: bool = True) -> pd.DataFrame:
    """Compute per-(ticker, fiscal_year) sentiment and risk rates.

    Sentiment is computed on Item 7 (MD&A); risk on Item 1A.
    Returns a DataFrame with columns:
        ticker, fiscal_year,
        sentiment_mda, risk_1a,
        words_mda, words_1a.
    Rows where the relevant section is missing/short get NaN.
    """
    if lm is None:
        lm = load_lm_dictionary()

    rows = []
    n = len(text_df)
    for i, r in enumerate(text_df.itertuples(index=False), 1):
        if verbose and i % 500 == 0:
            print(f"  scoring {i}/{n}")
        toks_mda = tokenise(getattr(r, "item_7", "") or "")
        toks_1a = tokenise(getattr(r, "item_1a", "") or "")
        s_mda = _score_tokens(toks_mda, lm)
        s_1a = _score_tokens(toks_1a, lm)
        rows.append({
            "ticker": r.ticker,
            "fiscal_year": int(r.fiscal_year),
            "sentiment_mda": s_mda["sentiment"],
            "risk_1a": s_1a["risk"],
            "words_mda": s_mda["n_words"],
            "words_1a": s_1a["n_words"],
        })
    return pd.DataFrame(rows)


# ── TF–IDF year-on-year similarity ────────────────────────────────────────

def compute_textsim(text_df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Compute YoY cosine similarity of (Item 1A + Item 7) TF–IDF vectors.

    A single corpus-wide TF–IDF is fitted on the union of all
    (ticker, fiscal_year) documents; for each firm-year the cosine is
    computed against the same firm's prior-year vector.  The first
    observation per firm has ``textsim = NaN``.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    df = text_df.copy()
    df["fiscal_year"] = df["fiscal_year"].astype(int)
    df["doc"] = (df["item_1a"].fillna("") + " " + df["item_7"].fillna("")).str.lower()
    df = df.sort_values(["ticker", "fiscal_year"]).reset_index(drop=True)

    if verbose:
        print(f"  fitting TF-IDF on {len(df)} documents …")

    vec = TfidfVectorizer(
        token_pattern=r"(?u)\b[a-z]{2,}\b",
        min_df=5,
        max_df=0.95,
        ngram_range=(1, 1),
        norm="l2",
        sublinear_tf=True,
        dtype=np.float32,
    )
    X = vec.fit_transform(df["doc"].tolist())

    sims = np.full(len(df), np.nan, dtype=np.float64)
    # Iterate within each firm in chronological order.
    for _ticker, idx in df.groupby("ticker").indices.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        # Years are already sorted within firm by the prior sort_values.
        for j in range(1, len(idx)):
            cur, prev = idx[j], idx[j - 1]
            sim = float(cosine_similarity(X[cur], X[prev])[0, 0])
            sims[cur] = sim

    out = df[["ticker", "fiscal_year"]].copy()
    out["textsim"] = sims
    return out


# ── Merge into the panel + first differences ──────────────────────────────

def merge_text_features(panel: pd.DataFrame,
                        scores: pd.DataFrame,
                        sims: pd.DataFrame) -> pd.DataFrame:
    """Merge raw text scores into the panel and add YoY first differences.

    Adds columns:
        sentiment_mda, risk_1a, words_mda, words_1a, textsim,
        delta_sentiment, delta_risk.
    """
    p = panel.copy()
    p = p.merge(scores, on=["ticker", "fiscal_year"], how="left")
    p = p.merge(sims, on=["ticker", "fiscal_year"], how="left")
    p = p.sort_values(["ticker", "fiscal_year"]).reset_index(drop=True)
    g = p.groupby("ticker", sort=False)
    p["delta_sentiment"] = g["sentiment_mda"].diff()
    p["delta_risk"] = g["risk_1a"].diff()
    return p


# ── Divergence ────────────────────────────────────────────────────────────

def calibrate_lambda(panel: pd.DataFrame,
                     dep: str = "delta_sentiment",
                     reg: str = "delta_roa") -> float:
    """Estimate λ as the within-firm OLS slope of *dep* on *reg*.

    Uses firm-demeaned variables, which is algebraically equivalent to
    OLS with firm fixed effects.  Returns NaN if there is insufficient
    within-firm variation.
    """
    df = panel[["ticker", dep, reg]].dropna()
    if df.empty:
        return float("nan")
    g = df.groupby("ticker")
    x = df[reg] - g[reg].transform("mean")
    y = df[dep] - g[dep].transform("mean")
    denom = float((x * x).sum())
    if denom <= 0:
        return float("nan")
    return float((x * y).sum() / denom)


def add_divergence(panel: pd.DataFrame, lambda_: float) -> pd.DataFrame:
    """Add ``divergence = delta_sentiment - lambda * delta_roa`` and the
    ``divergence_sign`` dummy = 1 if signs of Δsentiment and Δroa differ.
    """
    p = panel.copy()
    p["divergence"] = p["delta_sentiment"] - lambda_ * p["delta_roa"]
    sgn_s = np.sign(p["delta_sentiment"])
    sgn_r = np.sign(p["delta_roa"])
    valid = sgn_s.notna() & sgn_r.notna() & (sgn_s != 0) & (sgn_r != 0)
    p["divergence_sign"] = np.where(valid & (sgn_s != sgn_r), 1, 0)
    p.loc[~valid, "divergence_sign"] = np.nan
    return p


# ── Δroa helper (mirrors panel.py first-difference style) ─────────────────

def add_delta_roa(panel: pd.DataFrame) -> pd.DataFrame:
    """Add ``delta_roa`` as the within-firm year-on-year change in ROA."""
    p = panel.sort_values(["ticker", "fiscal_year"]).reset_index(drop=True)
    p["delta_roa"] = p.groupby("ticker", sort=False)["roa"].diff()
    return p


# ── Diagnostics output ────────────────────────────────────────────────────

def save_lambda_diagnostic(lambda_: float,
                           n_obs: int,
                           path: Path = DATA_DIR / "lambda.json") -> None:
    """Persist the calibrated λ and a small description for reproducibility."""
    payload = {
        "lambda": lambda_,
        "n_obs": int(n_obs),
        "spec": "within-firm OLS of delta_sentiment on delta_roa",
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  λ = {lambda_:.4f}  (n = {n_obs:,}) → {path.name}")

"""
The Silence Hypothesis: omission gap relative to SIC2-year peer union.

For each (sic2, fiscal_year) cell with at least MIN_PEERS firms:
  1. Build sum of TF-IDF row vectors across firms in the cell.
  2. For each focal firm i in the cell:
        peer_excl   = cell_sum - firm_i_vec        (leave-one-out)
        peer_dist   = peer_excl / peer_excl.sum()  (probability distribution
                                                    over peer disclosure mass)
        firm_terms  = (firm_i_vec > 0)
        OMISSION = sum_{t ∉ firm_terms} peer_dist[t]
                 = peer disclosure mass on terms the firm DOES NOT mention
        COMMISSION = sum_{t ∈ firm_terms, peer_dist[t] < eps} firm_dist[t]
                 = firm disclosure mass on terms peers DO NOT mention

Output: data/silence.csv with omission_1a, commission_1a, peer_count per
firm-year. Designed so the measure varies within firm-year by construction
because peer choices change every year.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MIN_PEERS = 5          # minimum cell size (excluding focal)
EPS = 1e-8             # zero threshold


def silence_for_section(df: pd.DataFrame, text_col: str, label: str):
    print(f"\n=== {label} ===")
    docs = df[text_col].fillna("").tolist()
    vec = TfidfVectorizer(
        lowercase=True, stop_words="english",
        min_df=5, max_df=0.95, dtype=np.float32,
    )
    X = vec.fit_transform(docs)        # (N, V) sparse, L2-normalised rows
    print(f"  matrix shape: {X.shape}, nnz={X.nnz:,}")

    # One row per (sic2, year) cell sum
    sic2 = (df["sic"] // 100).astype(int).values
    year = df["fiscal_year"].astype(int).values
    cells = pd.Series(list(zip(sic2, year)))
    cell_codes, cell_index = pd.factorize(cells)
    n_cells = len(cell_index)

    # Group sum: build (n_cells, V) sparse matrix of per-cell sums
    rows = sparse.coo_matrix(
        (np.ones(len(cell_codes), dtype=np.float32),
         (cell_codes, np.arange(len(cell_codes)))),
        shape=(n_cells, X.shape[0]),
    ).tocsr()
    cell_sums = rows @ X        # (n_cells, V)
    cell_counts = np.bincount(cell_codes)

    omission = np.full(X.shape[0], np.nan, dtype=np.float64)
    commission = np.full(X.shape[0], np.nan, dtype=np.float64)
    peer_n = np.zeros(X.shape[0], dtype=np.int32)

    for i in range(X.shape[0]):
        c = cell_codes[i]
        n_peers = cell_counts[c] - 1
        peer_n[i] = n_peers
        if n_peers < MIN_PEERS:
            continue
        firm_vec = X[i]                          # (1, V)
        peer_excl = cell_sums[c] - firm_vec      # (1, V)
        peer_total = peer_excl.sum()
        if peer_total <= 0:
            continue
        peer_dist = peer_excl / peer_total       # peer probability mass

        firm_mask = (firm_vec > EPS)             # bool sparse (1,V)
        # omission = peer mass on terms where firm has 0
        # = peer_total_mass - peer mass on terms firm DOES mention
        firm_idx = firm_mask.indices
        peer_on_firm_terms = float(peer_dist[0, firm_idx].sum())
        omission[i] = 1.0 - peer_on_firm_terms

        # commission = firm mass on terms where peer mass < eps
        firm_total = firm_vec.sum()
        if firm_total > 0:
            firm_dist = firm_vec / firm_total
            peer_on_firm_terms_raw = peer_excl[0, firm_idx].toarray().ravel()
            firm_on_firm_terms = firm_dist[0, firm_idx].toarray().ravel()
            novel = peer_on_firm_terms_raw < EPS
            commission[i] = float(firm_on_firm_terms[novel].sum())

    df_out = df[["ticker", "fiscal_year"]].copy()
    df_out[f"omission_{label}"] = omission
    df_out[f"commission_{label}"] = commission
    df_out[f"peer_n_{label}"] = peer_n

    # Stats
    valid = ~np.isnan(omission)
    print(f"  valid: {valid.sum():,}/{len(df):,}")
    print(f"  omission   mean={np.nanmean(omission):.3f}  "
          f"std={np.nanstd(omission):.3f}  "
          f"p25={np.nanpercentile(omission,25):.3f}  "
          f"p75={np.nanpercentile(omission,75):.3f}")
    print(f"  commission mean={np.nanmean(commission):.3f}  "
          f"std={np.nanstd(commission):.3f}")
    return df_out


def main():
    print("Reading text_sections.csv (this is large)...")
    sec = pd.read_csv(DATA / "text_sections.csv")
    print(f"  {len(sec):,} firm-years")

    if "sic" not in sec.columns:
        # Bring sic from annual_panel_text.csv
        panel = pd.read_csv(DATA / "annual_panel_text.csv",
                              usecols=["ticker", "fiscal_year", "sic"])
        sec = sec.merge(panel, on=["ticker", "fiscal_year"], how="left")

    sec = sec.dropna(subset=["sic"]).copy()
    sec["sic"] = sec["sic"].astype(int)

    out_1a = silence_for_section(sec, "item_1a", "1a")
    out_7 = silence_for_section(sec, "item_7", "7")

    out = out_1a.merge(out_7, on=["ticker", "fiscal_year"], how="outer")
    path = DATA / "silence.csv"
    out.to_csv(path, index=False)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()

"""
The Silence Hypothesis: point-in-time omission gap relative to SIC2 peers.

Everything used for a focal filing (firm i, fiscal year t, filed on day f_i)
was public before f_i:

  * Peers.  For every other panel firm k in the same SIC2 industry we take
    its most recent Item 1A filed STRICTLY BEFORE f_i, provided it belongs
    to fiscal year t or t-1 (so it describes the same period or the one
    just before).  Peers that file after the focal firm are never used.
  * Vocabulary and IDF weights.  For fiscal year t the TF-IDF model is
    fitted only on Item 1A documents filed before the first fiscal-year-t
    filing in the panel.  Fiscal year 2010 has no earlier documents in the
    panel, so its omission gap is missing.

Then, as before:
    peer_excl  = sum of peer TF-IDF rows (focal firm never included)
    peer_dist  = peer_excl / peer_excl.sum()
    OMISSION   = 1 - peer mass on terms the focal firm mentions
    COMMISSION = focal mass on terms no peer mentions
A focal filing needs at least MIN_PEERS peers.

Output: data/silence.csv with omission_*, commission_*, peer_n_* per
firm-year.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MIN_PEERS = 5          # minimum number of peers (focal firm excluded)
EPS = 1e-8             # zero threshold


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, stop_words="english",
                           min_df=5, max_df=0.95, dtype=np.float32)


def silence_for_section(df: pd.DataFrame, text_col: str, label: str):
    """Compute the point-in-time omission / commission gap.

    *df* needs columns ticker, fiscal_year, sic, filing_date, *text_col*.
    """
    print(f"\n=== {label} ===")
    df = df.reset_index(drop=True).copy()
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    df["fiscal_year"] = df["fiscal_year"].astype(int)
    df["sic2"] = (df["sic"] // 100).astype(int)
    docs = df[text_col].fillna("")
    has_doc = docs.str.len() > 0

    n = len(df)
    omission = np.full(n, np.nan)
    commission = np.full(n, np.nan)
    peer_n = np.zeros(n, dtype=np.int32)

    for t in sorted(df["fiscal_year"].unique()):
        focal_idx = np.where(df["fiscal_year"].to_numpy() == t)[0]
        first_fd = df.loc[focal_idx, "filing_date"].min()
        train = df.index[(df["filing_date"] < first_fd) & has_doc]
        if len(train) < 50:
            print(f"  FY{t}: only {len(train)} earlier documents -> skipped")
            continue
        vec = _vectorizer().fit(docs[train])

        # Candidate peer documents: fiscal years t and t-1.
        cand = df.index[df["fiscal_year"].isin([t, t - 1]) & has_doc]
        rows = np.concatenate([focal_idx, cand.to_numpy()])
        rows = np.unique(rows)
        X = vec.transform(docs[rows])
        pos = {r: j for j, r in enumerate(rows)}

        cand_df = df.loc[cand, ["ticker", "sic2", "filing_date"]]
        for i in focal_idx:
            fd_i = df.at[i, "filing_date"]
            peers = cand_df[(cand_df["sic2"] == df.at[i, "sic2"])
                            & (cand_df["ticker"] != df.at[i, "ticker"])
                            & (cand_df["filing_date"] < fd_i)]
            # most recent eligible filing per peer firm
            peers = peers.sort_values("filing_date").drop_duplicates(
                "ticker", keep="last")
            peer_n[i] = len(peers)
            if len(peers) < MIN_PEERS or not has_doc[i]:
                continue
            firm_vec = X[pos[i]]
            peer_excl = np.asarray(
                X[[pos[k] for k in peers.index]].sum(axis=0)).ravel()
            peer_total = peer_excl.sum()
            if peer_total <= 0:
                continue
            keep = firm_vec.data > EPS
            firm_idx = firm_vec.indices[keep]
            firm_val = firm_vec.data[keep]
            omission[i] = 1.0 - float(peer_excl[firm_idx].sum()) / peer_total

            firm_total = firm_val.sum()
            if firm_total > 0:
                novel = peer_excl[firm_idx] < EPS
                commission[i] = float(firm_val[novel].sum() / firm_total)
        ok = np.isfinite(omission[focal_idx]).sum()
        print(f"  FY{t}: vocab={len(vec.vocabulary_):,}  "
              f"train docs={len(train):,}  valid={ok}/{len(focal_idx)}")

    df_out = df[["ticker", "fiscal_year"]].copy()
    df_out[f"omission_{label}"] = omission
    df_out[f"commission_{label}"] = commission
    df_out[f"peer_n_{label}"] = peer_n

    valid = ~np.isnan(omission)
    print(f"  valid: {valid.sum():,}/{n:,}")
    print(f"  omission   mean={np.nanmean(omission):.3f}  "
          f"std={np.nanstd(omission):.3f}  "
          f"p25={np.nanpercentile(omission, 25):.3f}  "
          f"p75={np.nanpercentile(omission, 75):.3f}")
    print(f"  commission mean={np.nanmean(commission):.3f}  "
          f"std={np.nanstd(commission):.3f}")
    return df_out


def load_sections(cols=("item_1a", "item_7")) -> pd.DataFrame:
    """Text sections joined with sic and filing_date from the panel."""
    sec = pd.read_csv(DATA / "text_sections.csv",
                      usecols=["ticker", "fiscal_year", *cols])
    panel = pd.read_csv(DATA / "annual_panel_text.csv",
                        usecols=["ticker", "fiscal_year", "sic", "filing_date"])
    sec = sec.merge(panel, on=["ticker", "fiscal_year"], how="inner")
    sec = sec.dropna(subset=["sic", "filing_date"]).copy()
    sec["sic"] = sec["sic"].astype(int)
    return sec


def main():
    print("Reading text_sections.csv (this is large)...")
    sec = load_sections()
    print(f"  {len(sec):,} firm-years")

    out_1a = silence_for_section(sec, "item_1a", "1a")
    out_7 = silence_for_section(sec, "item_7", "7")

    out = out_1a.merge(out_7, on=["ticker", "fiscal_year"], how="outer")
    path = DATA / "silence.csv"
    out.to_csv(path, index=False)
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()

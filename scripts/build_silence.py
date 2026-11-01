"""
H3 omission gap of Item 1A relative to SIC2 peers (point-in-time).

Everything used for a focal filing (firm i, fiscal year t, filed on day f_i)
was public before f_i:

  * Peers.  For every other panel firm k in the same SIC2 cell we take its
    most recent Item 1A filed STRICTLY BEFORE f_i, provided it belongs to
    fiscal year t or t-1.  Peers that file after the focal firm are never used.
  * Vocabulary and IDF weights.  For fiscal year t the TF-IDF model is fitted
    only on Item 1A documents filed before the first fiscal-year-t filing in
    the panel (FY2010 therefore has no omission gap).

    peer_agg   = sum of peer TF-IDF rows (focal firm never included)
    OMISSION   = 1 - (peer mass on terms the focal firm uses) / (total peer mass)
    COMMISSION = focal mass on terms no peer uses
A focal filing needs at least MIN_PEERS peers.

TF-IDF settings (#39): scikit-learn TfidfVectorizer, lowercase, token pattern
(?u)\\b\\w\\w+\\b (runs of >= 2 letters/digits/underscores), scikit-learn
English stop-word list removed, unigrams, min_df=5, max_df=0.95, raw term
counts (sublinear_tf=False), smoothed IDF, rows l2-normalised; vocabulary and
IDF re-fitted for every fiscal year on earlier filings only.

Variants written alongside the headline measure (all from stored text only):
  omission_1a            industry = historical SIC2 at filing (default, #72)
  omission_1a_sicnow     industry = today's SIC2 snapshot (pre-revision measure)
  omission_pooled_1a     one pooled vocabulary/IDF fitted on all years (#37);
                         uses later documents, so only a diagnostic
  omission_fixedlen_1a   length-invariant version (#38, #64): the focal firm's
                         term set is taken from K random contiguous blocks of
                         BLOCK_LEN raw tokens of its own Item 1A, omission is
                         computed per block and averaged; filings shorter than
                         BLOCK_LEN use the whole text (fixedlen_short_1a = 1)
Δomission decomposition (#37) -> data/silence_decomp.csv and
data/results/omission_decomposition.csv.

Output: data/silence.csv
"""
from __future__ import annotations

import pickle
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from text_features import load_text_sections, rekey_text

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CACHE = Path.home() / "scratch" / "bak" / "silence_cache"

MIN_PEERS = 5
MIN_TRAIN = 50
EPS = 1e-8
BLOCK_LEN = 3000
N_BLOCKS = 20
SEED = 20261101


def _vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, stop_words="english",
                           min_df=5, max_df=0.95, dtype=np.float32)


# ── data ─────────────────────────────────────────────────────────────────

def load_sections(col: str = "item_1a") -> pd.DataFrame:
    """Stored text joined with SIC (today + historical) and filing date."""
    pk = CACHE.parent / "text_sections.pkl"
    src = DATA / "text_sections.csv"
    if pk.exists() and pk.stat().st_mtime >= src.stat().st_mtime:
        sec = rekey_text(pd.read_pickle(pk)[["ticker", "fiscal_year", col]])
    else:
        sec = load_text_sections((col,))
    pcols = ["ticker", "fiscal_year", "sic", "filing_date"]
    panel = pd.read_csv(DATA / "annual_panel_text.csv", low_memory=False)
    if "sic_hist" in panel.columns:
        pcols.append("sic_hist")
    sec = sec.merge(panel[pcols], on=["ticker", "fiscal_year"], how="inner")
    sec = sec.dropna(subset=["sic", "filing_date"]).copy()
    if "sic_hist" not in sec.columns:
        sec["sic_hist"] = sec["sic"]
    sec["sic_hist"] = sec["sic_hist"].fillna(sec["sic"])
    sec["filing_date"] = pd.to_datetime(sec["filing_date"])
    sec["fiscal_year"] = sec["fiscal_year"].astype(int)
    sec["doc"] = sec[col].fillna("")
    sec["has_doc"] = sec["doc"].str.len() > 0
    return sec.drop(columns=[col]).sort_values(["fiscal_year", "ticker"]).reset_index(drop=True)


# ── TF-IDF spaces (cached; independent of the peer definition) ───────────

def prepare_spaces(sec: pd.DataFrame, pooled: bool = False, verbose=True) -> dict:
    """{t: dict(rows, X, vocab, vec)} with rows = focal(t) ∪ docs of t and t-1."""
    sig = "|".join(f"{a}:{b}:{c}:{len(d)}" for a, b, c, d in
                   zip(sec["ticker"], sec["fiscal_year"], sec["filing_date"], sec["doc"]))
    key = f"{'pooled' if pooled else 'pit'}_{zlib.crc32(sig.encode()):08x}"
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"spaces_{key}.pkl"
    if path.exists():
        with open(path, "rb") as f:
            return pickle.load(f)
    docs, has = sec["doc"], sec["has_doc"].to_numpy()
    fy, fd = sec["fiscal_year"].to_numpy(), sec["filing_date"]
    spaces = {}
    pooled_vec = _vectorizer().fit(docs[has]) if pooled else None
    for t in sorted(np.unique(fy)):
        focal = np.where(fy == t)[0]
        if pooled:
            vec = pooled_vec
        else:
            train = np.where((fd < fd.iloc[focal].min()).to_numpy() & has)[0]
            if len(train) < MIN_TRAIN:
                if verbose:
                    print(f"  FY{t}: only {len(train)} earlier documents -> skipped")
                continue
            vec = _vectorizer().fit(docs.iloc[train])
        rows = np.unique(np.concatenate([focal, np.where(np.isin(fy, [t, t - 1]) & has)[0]]))
        X = vec.transform(docs.iloc[rows]).tocsr()
        spaces[t] = {"rows": rows, "X": X, "vocab": vec.vocabulary_}
        if verbose:
            print(f"  FY{t}: vocab={len(vec.vocabulary_):,} rows={len(rows)}")
    with open(path, "wb") as f:
        pickle.dump(spaces, f, protocol=pickle.HIGHEST_PROTOCOL)
    return spaces


# ── peer sets and omission ───────────────────────────────────────────────

def peer_sets(sec: pd.DataFrame, spaces: dict, cell_of) -> dict:
    """{t: (focal_rows, list of peer row-positions in spaces[t]['rows'])}.

    *cell_of(t)* returns a Series ticker -> cell id used in year t."""
    out = {}
    tick = sec["ticker"].to_numpy()
    fy = sec["fiscal_year"].to_numpy()
    fd = sec["filing_date"].to_numpy()
    has = sec["has_doc"].to_numpy()
    for t, S in spaces.items():
        rows = S["rows"]
        cell = cell_of(t)
        rcell = pd.Series(tick[rows]).map(cell).to_numpy()
        cand = has[rows]
        focal = rows[fy[rows] == t]
        pos = {r: j for j, r in enumerate(rows)}
        sets = []
        order = np.argsort(fd[rows], kind="stable")
        for i in focal:
            c = cell.get(tick[i], np.nan)
            m = cand & (rcell == c) & (tick[rows] != tick[i]) & (fd[rows] < fd[i])
            js = order[m[order]]                 # eligible, by filing date
            last = {}
            for j in js:                         # most recent per peer firm
                last[tick[rows[j]]] = j
            sets.append(np.fromiter(last.values(), dtype=np.int64))
        out[t] = (focal, sets, pos)
    return out


def _peer_matrix(sets, n_rows) -> sp.csr_matrix:
    indptr = np.cumsum([0] + [len(s) for s in sets])
    idx = np.concatenate(sets) if len(sets) else np.array([], dtype=np.int64)
    return sp.csr_matrix((np.ones(len(idx), np.float32), idx, indptr),
                         shape=(len(sets), n_rows))


def omission_from_sets(sec, spaces, psets, n_total: int):
    om = np.full(n_total, np.nan)
    com = np.full(n_total, np.nan)
    pn = np.zeros(n_total, dtype=np.int32)
    has = sec["has_doc"].to_numpy()
    for t, (focal, sets, pos) in psets.items():
        X = spaces[t]["X"]
        P = _peer_matrix(sets, X.shape[0]) @ X            # peer aggregates
        F = X[[pos[i] for i in focal]]
        B = F.copy(); B.data = (B.data > EPS).astype(np.float32)
        tot = np.asarray(P.sum(axis=1)).ravel()
        cov = np.asarray(B.multiply(P).sum(axis=1)).ravel()
        Pb = P.copy(); Pb.data = (Pb.data > EPS).astype(np.float32)
        fin = np.asarray(F.multiply(Pb).sum(axis=1)).ravel()
        ftot = np.asarray(F.sum(axis=1)).ravel()
        n = np.array([len(s) for s in sets])
        ok = (n >= MIN_PEERS) & has[focal] & (tot > 0)
        pn[focal] = n
        om[focal[ok]] = 1.0 - cov[ok] / tot[ok]
        okc = ok & (ftot > 0)
        com[focal[okc]] = (ftot[okc] - fin[okc]) / ftot[okc]
    return om, com, pn


def fixedlen_omission(sec, spaces, psets, n_total: int):
    """Average omission over N_BLOCKS random contiguous BLOCK_LEN-token
    blocks of the focal Item 1A (peer aggregate unchanged)."""
    tok = _vectorizer().build_tokenizer()
    pre = _vectorizer().build_preprocessor()
    om = np.full(n_total, np.nan)
    short = np.full(n_total, np.nan)
    tick = sec["ticker"].to_numpy()
    for t, (focal, sets, pos) in psets.items():
        X, vocab = spaces[t]["X"], spaces[t]["vocab"]
        P = (_peer_matrix(sets, X.shape[0]) @ X).tocsr()
        tot = np.asarray(P.sum(axis=1)).ravel()
        for k, i in enumerate(focal):
            if len(sets[k]) < MIN_PEERS or tot[k] <= 0 or not sec.at[i, "has_doc"]:
                continue
            ids = np.fromiter((vocab.get(w, -1) for w in tok(pre(sec.at[i, "doc"]))),
                              dtype=np.int64)
            prow = P.getrow(k).toarray().ravel()
            n = len(ids)
            if n <= BLOCK_LEN:
                blocks = [ids]
                short[i] = 1
            else:
                rng = np.random.default_rng([SEED, int(t), zlib.crc32(tick[i].encode())])
                starts = rng.integers(0, n - BLOCK_LEN + 1, size=N_BLOCKS)
                blocks = [ids[s:s + BLOCK_LEN] for s in starts]
                short[i] = 0
            vals = []
            for b in blocks:
                u = np.unique(b[b >= 0])
                vals.append(1.0 - prow[u].sum() / tot[k])
            om[i] = float(np.mean(vals))
        print(f"  fixed-length FY{t}: done ({np.isfinite(om[focal]).sum()} valid)")
    return om, short


def cells_by(sec: pd.DataFrame, col: str):
    """cell_of(t): ticker -> SIC2 from *col* in fiscal year t (falls back to
    the firm's t-1 code for firms that only have a t-1 document)."""
    s = sec[["ticker", "fiscal_year", col]].copy()
    s["c"] = (s[col] // 100).astype(int)
    by = {t: g.set_index("ticker")["c"] for t, g in s.groupby("fiscal_year")}

    def cell_of(t):
        cur = by.get(t, pd.Series(dtype=int))
        prev = by.get(t - 1, pd.Series(dtype=int))
        return pd.concat([cur, prev[~prev.index.isin(cur.index)]]).to_dict()
    return cell_of


# ── Δomission decomposition (#37) ────────────────────────────────────────

def _om(f_row, peer_rows):
    if peer_rows.shape[0] < MIN_PEERS:
        return np.nan
    p = np.asarray(peer_rows.sum(axis=0)).ravel()
    tot = p.sum()
    if tot <= 0:
        return np.nan
    return 1.0 - p[f_row.indices[f_row.data > EPS]].sum() / tot


def decompose(sec, spaces, cell_of):
    """Year-t vocabulary; FY documents (not point-in-time).  For firm i:
       Δ = Ω(o_t, C_t) − Ω(o_{t-1}, C_{t-1})
       own   = Ω(o_t, C_{t-1}) − Ω(o_{t-1}, C_{t-1})
       peer  = Ω(o_{t-1}, S_t) − Ω(o_{t-1}, S_{t-1})   S = peers in both years
       comp  = [Ω(o_{t-1}, C_t) − Ω(o_{t-1}, S_t)] − [Ω(o_{t-1}, C_{t-1}) − Ω(o_{t-1}, S_{t-1})]
       inter = Δ − own − peer − comp
    C_t / C_{t-1}: cell members with an FY t / t-1 document (texts of that year)."""
    tick = sec["ticker"].to_numpy()
    fy = sec["fiscal_year"].to_numpy()
    out = []
    for t, S in spaces.items():
        rows, X = S["rows"], S["X"]
        if not (fy[rows] == t - 1).any():
            continue
        has = sec["has_doc"].to_numpy()[rows]
        cur, prev = cell_of(t), cell_of(t - 1)
        doc_t = {tick[r]: j for j, r in enumerate(rows) if fy[r] == t and has[j]}
        doc_p = {tick[r]: j for j, r in enumerate(rows) if fy[r] == t - 1 and has[j]}
        for f in doc_t:
            if f not in doc_p or f not in cur or f not in prev:
                continue
            Ct = [doc_t[k] for k in doc_t if k != f and cur.get(k) == cur[f]]
            Cp = [doc_p[k] for k in doc_p if k != f and prev.get(k) == prev[f]]
            both = [k for k in doc_t if k != f and cur.get(k) == cur[f]
                    and k in doc_p and prev.get(k) == prev[f]]
            St = [doc_t[k] for k in both]
            Sp = [doc_p[k] for k in both]
            ot, op = X[doc_t[f]], X[doc_p[f]]
            a = _om(ot, X[Ct]); b = _om(op, X[Cp]); c = _om(ot, X[Cp])
            d = _om(op, X[St]); e = _om(op, X[Sp]); g = _om(op, X[Ct])
            rec = {"ticker": f, "fiscal_year": int(t), "delta": a - b, "own": c - b,
                   "peer_text": d - e, "composition": (g - d) - (b - e),
                   "n_Ct": len(Ct), "n_Cp": len(Cp), "n_stay": len(both)}
            rec["interaction"] = rec["delta"] - rec["own"] - rec["peer_text"] - rec["composition"]
            out.append(rec)
    return pd.DataFrame(out)


def decomposition_summary(dec: pd.DataFrame) -> pd.DataFrame:
    d = dec.dropna()
    v = d["delta"].var()
    rows = []
    for c in ("own", "peer_text", "composition", "interaction"):
        rows.append({"component": c, "n": len(d), "sd": d[c].std(),
                     "var_share": d[c].var() / v,
                     "cov_share": np.cov(d[c], d["delta"])[0, 1] / v})
    rows.append({"component": "delta", "n": len(d), "sd": d["delta"].std(),
                 "var_share": 1.0, "cov_share": 1.0})
    return pd.DataFrame(rows)


# ── main ─────────────────────────────────────────────────────────────────

def main():
    print("Reading stored Item 1A text …")
    sec = load_sections("item_1a")
    n = len(sec)
    print(f"  {n:,} firm-years, {int(sec['has_doc'].sum()):,} with text")

    print("\nTF-IDF spaces (point-in-time) …")
    spaces = prepare_spaces(sec)
    out = sec[["ticker", "fiscal_year"]].copy()

    for col, suffix in (("sic_hist", ""), ("sic", "_sicnow")):
        ps = peer_sets(sec, spaces, cells_by(sec, col))
        om, com, pn = omission_from_sets(sec, spaces, ps, n)
        out[f"omission_1a{suffix}"] = om
        out[f"commission_1a{suffix}"] = com
        out[f"peer_n_1a{suffix}"] = pn
        print(f"  {col}: valid {np.isfinite(om).sum():,}  mean={np.nanmean(om):.3f}")
        if col == "sic_hist":
            ps_hist = ps

    print("\nFixed-length omission …")
    fl, short = fixedlen_omission(sec, spaces, ps_hist, n)
    out["omission_fixedlen_1a"] = fl
    out["fixedlen_short_1a"] = short

    print("\nPooled-vocabulary omission …")
    pooled = prepare_spaces(sec, pooled=True)
    ps_pool = peer_sets(sec, pooled, cells_by(sec, "sic_hist"))
    out["omission_pooled_1a"], _, _ = omission_from_sets(sec, pooled, ps_pool, n)

    out.to_csv(DATA / "silence.csv", index=False)
    print(f"\nWrote {DATA / 'silence.csv'}")

    print("\nΔomission decomposition …")
    dec = decompose(sec, spaces, cells_by(sec, "sic_hist"))
    dec.to_csv(DATA / "silence_decomp.csv", index=False)
    summ = decomposition_summary(dec)
    (DATA / "results").mkdir(exist_ok=True)
    summ.to_csv(DATA / "results" / "omission_decomposition.csv", index=False)
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

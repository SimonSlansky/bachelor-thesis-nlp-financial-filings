"""Estimation for H1/H2 and the shared infrastructure used by every model.

Everything that estimates a regression imports from here (the H3 scripts
included), so the control vector, the sample rules, the FE/cluster choices
and the results format are identical across hypotheses (#41).

  CONTROLS          common control vector (Phase-0 decision 2)
  load_text_panel   one loader: liquidity rule, text-mismatch rule,
                    horizon-specific outcome, industry definition,
                    regression-sample winsorisation
  fit               AbsorbingLS with absorbed FE and clustered SEs; adds the
                    within-R^2
  tidy / save       one results store: data/results/<name>.csv (long format)

The LaTeX tables are written by make_tables.py from data/results/.

Run:  python scripts/regressions.py [--only main horizons ...] [--boot B]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv.absorbing import AbsorbingLS

from config import DATA_DIR
from panel import winsorize_ratios, WINSORIZE_COLS, consecutive_lag

RESULTS_DIR = DATA_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# ── Variables ────────────────────────────────────────────────────────────

CONTROLS = ["pre_log_vol", "log_mcap", "leverage", "roa", "asset_growth",
            "bm", "ret_pre_12m", "current_ratio", "ocf_to_assets"]

LEN_VAR, DENS_VAR, UNC_VAR, LIT_VAR = "log_words_1a", "risk_1a", "unc_1a", "lit_1a"
TEXT_VARS = [LEN_VAR, DENS_VAR, UNC_VAR, LIT_VAR]
COUNT_VARS = [LEN_VAR, "log_n_unc_1a", "log_n_lit_1a"]

# Pre-revision specification, kept only for old-vs-new comparisons.
ORIG_CONTROLS = ["lagged_log_vol", "log_total_assets", "leverage", "roa",
                 "asset_growth"]

HORIZONS = ["5d", "10d", "30d", "90d", "180d", "365d"]
HEADLINE = "30d"

SPEC_COLUMNS = {           # the five nested columns of the headline table
    "(1)": [],
    "(2)": [LEN_VAR],
    "(3)": [DENS_VAR],
    "(4)": [LEN_VAR, DENS_VAR],
    "(5)": [LEN_VAR, UNC_VAR, LIT_VAR],
}
DECOMP = SPEC_COLUMNS["(5)"]
JOINT = SPEC_COLUMNS["(4)"]

# ── Loading ──────────────────────────────────────────────────────────────

_CACHE: dict[str, pd.DataFrame] = {}


def _read_panel() -> pd.DataFrame:
    if "panel" not in _CACHE:
        p = pd.read_csv(DATA_DIR / "annual_panel_text.csv", low_memory=False)
        sil = DATA_DIR / "silence.csv"
        if sil.exists():
            s = pd.read_csv(sil)
            p = p.drop(columns=[c for c in s.columns
                                if c in p.columns and c not in ("ticker", "fiscal_year")])
            p = p.merge(s, on=["ticker", "fiscal_year"], how="left")
        acc = DATA_DIR / "accession_changes.csv"
        if acc.exists():
            a = pd.read_csv(acc)[["ticker", "fiscal_year", "text_mismatch"]]
            p = p.drop(columns=[c for c in ("text_mismatch",) if c in p.columns])
            p = p.merge(a, on=["ticker", "fiscal_year"], how="left")
        p["text_mismatch"] = p.get("text_mismatch", 0)
        p["text_mismatch"] = pd.to_numeric(p["text_mismatch"], errors="coerce").fillna(0)
        _CACHE["panel"] = p
    return _CACHE["panel"].copy()


FF48_FILE = DATA_DIR / "external" / "Siccodes48.txt"


def ff48_map() -> dict[int, int]:
    """SIC4 -> Fama-French 48 industry (Ken French data library,
    Siccodes48.txt); SIC codes outside every range -> 48 (Other)."""
    import re
    out, ind = {}, None
    for line in FF48_FILE.read_text(encoding="latin-1").splitlines():
        m = re.match(r"\s*(\d+)\s+\w+", line)
        r = re.match(r"\s*(\d{4})-(\d{4})", line)
        if r:
            for s in range(int(r.group(1)), int(r.group(2)) + 1):
                out.setdefault(s, ind)
        elif m:
            ind = int(m.group(1))
    return out


def industry_code(df: pd.DataFrame, ind: str) -> pd.Series:
    """Industry identifier for the FE.  Default: historical SIC2 (#72)."""
    hist = df["sic_hist"] if "sic_hist" in df.columns else df["sic"]
    if ind == "sic2_hist":
        return hist // 100
    if ind == "sic2":                      # today's SIC snapshot
        return df["sic"] // 100
    if ind == "sic3":
        return hist // 10
    if ind == "ff48":
        mp = ff48_map()
        return hist.map(lambda s: mp.get(int(s), 48) if pd.notna(s) else np.nan)
    raise ValueError(ind)


def _log_pos(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return np.log(s.where(s > 0))


def _bool(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def load_text_panel(h: str = HEADLINE, ind: str = "sic2_hist",
                    xvars: list[str] | None = None, extra: tuple = (),
                    dv: str = "vol", liquid: bool = True,
                    drop_mismatch: bool = True, winsor: bool = True
                    ) -> pd.DataFrame:
    """Regression sample for horizon *h* (e.g. "30d").

    Adds  log_vol (DV: ln vol_h; dv="idio": ln idio_vol_h; dv="exea": ln
          vol_h without earnings-release reaction days),
          pre_log_vol, lagged_log_vol, log_mkt_vol, ea, ind, fm_year, post2019.
    Keeps rows with every variable in *xvars* (default CONTROLS+TEXT_VARS)
    and *extra* non-missing; winsorises WINSORIZE_COLS on that sample.
    """
    df = _read_panel()
    vcol = {"vol": f"vol_{h}", "idio": f"idio_vol_{h}", "exea": f"vol_{h}_exea"}[dv]
    df["log_vol"] = _log_pos(df[vcol])
    df["pre_log_vol"] = _log_pos(df["pre_vol"])
    df["lagged_log_vol"] = _log_pos(df.get(f"lagged_vol_{h}", np.nan))
    df["log_mkt_vol"] = _log_pos(df.get(f"mkt_vol_{h}", np.nan))
    df["ea"] = pd.to_numeric(df.get(f"ea_in_{h}", np.nan), errors="coerce")
    df["ind"] = industry_code(df, ind)
    df["fm_year"] = (pd.to_numeric(df["filing_month"], errors="coerce") * 10000
                     + df["fiscal_year"])
    df["post2019"] = (df["fiscal_year"] >= 2020).astype(float)
    if liquid:
        df = df[_bool(df["liquid_pre"])]
    if drop_mismatch:
        df = df[df["text_mismatch"] != 1]
    xvars = list(CONTROLS + TEXT_VARS) if xvars is None else list(xvars)
    need = list(dict.fromkeys(["log_vol", "ind", "fiscal_year", "ticker",
                               *xvars, *extra]))
    df = df.dropna(subset=need)
    if winsor:
        df = winsorize_ratios(df, cols=[c for c in WINSORIZE_COLS if c in df.columns])
    return df.reset_index(drop=True)


# ── Estimation ───────────────────────────────────────────────────────────

def fit(df: pd.DataFrame, xvars: list[str], absorb=("ind", "fiscal_year"),
        cluster=("ticker", "fiscal_year"), y: str = "log_vol"):
    """AbsorbingLS: FE in *absorb* absorbed, SEs clustered on *cluster*
    (one- or two-way; None = heteroskedasticity-robust).  Adds
    ``res.r2_within`` = 1 - SSR / SSR(y on the absorbed FE only)."""
    d = df.dropna(subset=[y, *xvars, *absorb])
    X = d[xvars].astype(float)
    X.insert(0, "const", 1.0)
    A = pd.DataFrame({f"fe_{a}": d[a].astype(str).astype("category") for a in absorb},
                     index=d.index)
    mod = AbsorbingLS(d[y].astype(float), X, absorb=A, drop_absorbed=True)
    if cluster:
        cl = pd.DataFrame({c: d[c].astype("category").cat.codes for c in cluster},
                          index=d.index)
        res = mod.fit(cov_type="clustered", clusters=cl)
    else:
        res = mod.fit(cov_type="robust")
    y_w = np.asarray(mod.absorbed_dependent).ravel()
    res.r2_within = float(1 - (np.asarray(res.resids) ** 2).sum() / (y_w ** 2).sum())
    res.n_firms = int(d["ticker"].nunique())
    res.sample = d
    return res


def tidy(res, model: str, column: str = "", **meta) -> list[dict]:
    rows = []
    for v in res.params.index:
        if v == "const":
            continue
        rows.append({"model": model, "column": column, "var": v,
                     "coef": float(res.params[v]), "se": float(res.std_errors[v]),
                     "t": float(res.tstats[v]), "p": float(res.pvalues[v]),
                     "n": int(res.nobs), "firms": res.n_firms,
                     "r2_within": res.r2_within, "r2": float(res.rsquared),
                     "r2_adj": float(res.rsquared_adj), **meta})
    return rows


def save(rows: list[dict], name: str) -> pd.DataFrame:
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / f"{name}.csv", index=False)
    print(f"  -> data/results/{name}.csv ({len(out)} rows)")
    return out


def wald_equal(res, a: str, b: str) -> dict:
    from scipy.stats import norm
    c = res.cov
    var = c.loc[a, a] + c.loc[b, b] - 2 * c.loc[a, b]
    z = float((res.params[a] - res.params[b]) / np.sqrt(var))
    return {"diff": float(res.params[a] - res.params[b]), "z": z,
            "p": float(2 * (1 - norm.cdf(abs(z))))}


def _show(res, vars_: list[str], label: str = "") -> None:
    print(f"  {label}  N={int(res.nobs):,} firms={res.n_firms} "
          f"R2w={res.r2_within:.3f}")
    for v in vars_:
        if v in res.params.index:
            print(f"     {v:16s} {res.params[v]:+9.4f}  t={res.tstats[v]:+6.2f}  "
                  f"p={res.pvalues[v]:.4f}")


# ── H1/H2 analyses (§6.1) ────────────────────────────────────────────────

def run_main(h: str = HEADLINE) -> pd.DataFrame:
    """Five nested columns on one common sample; incremental within-R^2."""
    df = load_text_panel(h)
    rows, base_r2 = [], None
    for col, text in SPEC_COLUMNS.items():
        r = fit(df, CONTROLS + text)
        base_r2 = r.r2_within if base_r2 is None else base_r2
        rows += tidy(r, f"main_{h}", col, h=h, absorb="ind+year",
                     cluster="firm+year", incr_r2_within=r.r2_within - base_r2)
        _show(r, text, f"main {h} {col}")
        if col == "(5)":
            w = wald_equal(r, UNC_VAR, LIT_VAR)
            rows.append({"model": f"main_{h}", "column": col, "var": "wald_unc_eq_lit",
                         "coef": w["diff"], "t": w["z"], "p": w["p"],
                         "n": int(r.nobs), "firms": r.n_firms, "h": h})
    return save(rows, f"main_{h}")


def run_horizons() -> pd.DataFrame:
    rows = []
    for h in HORIZONS:
        df = load_text_panel(h)
        for col in ("(4)", "(5)"):
            r = fit(df, CONTROLS + SPEC_COLUMNS[col])
            rows += tidy(r, "horizons", col, h=h, absorb="ind+year", cluster="firm+year")
        _show(r, DECOMP, f"horizon {h}")
    return save(rows, "horizons")


def run_firm_fe(h: str = HEADLINE) -> pd.DataFrame:
    """Firm+year FE without the lagged DV (pre_log_vol is the dynamic
    control); the LDV version is reported for the Nickell comparison (#44)."""
    rows = []
    df = load_text_panel(h, extra=("lagged_log_vol",))
    for col in ("(4)", "(5)"):
        for tag, xs in (("no_ldv", CONTROLS), ("with_ldv", CONTROLS + ["lagged_log_vol"])):
            for absorb, an in ((("ticker", "fiscal_year"), "firm+year"),
                               (("ind", "fiscal_year"), "ind+year")):
                r = fit(df, xs + SPEC_COLUMNS[col], absorb=absorb)
                rows += tidy(r, "firm_fe", f"{col}_{tag}_{an}", h=h, absorb=an,
                             cluster="firm+year", ldv=tag == "with_ldv")
                if col == "(5)":
                    _show(r, DECOMP + ["lagged_log_vol"], f"{an} {tag}")
    return save(rows, "firm_fe")


def _timing_panel(h: str) -> pd.DataFrame:
    df = load_text_panel(h, xvars=CONTROLS)
    full = _read_panel()[["ticker", "fiscal_year", *TEXT_VARS]]
    full = full.sort_values(["ticker", "fiscal_year"]).reset_index(drop=True)
    for v in TEXT_VARS:
        full[f"{v}_lag1"] = consecutive_lag(full, v)
    df = df.drop(columns=[f"{v}_lag1" for v in TEXT_VARS if f"{v}_lag1" in df])
    df = df.merge(full[["ticker", "fiscal_year", *[f"{v}_lag1" for v in TEXT_VARS]]],
                  on=["ticker", "fiscal_year"], how="left")
    for v in TEXT_VARS:
        df[f"d_{v}"] = df[v] - df[f"{v}_lag1"]
    return df


def run_robustness(h: str = HEADLINE) -> pd.DataFrame:
    """Every H1/H2 robustness model, decomposition spec (col 5) unless noted."""
    rows = []
    X5 = CONTROLS + DECOMP

    def add(name, r, **m):
        rows.extend(tidy(r, "robustness", name, h=h, **m))
        _show(r, [v for v in r.params.index if v in DECOMP + COUNT_VARS
                  or v.endswith("_x_post") or v.startswith("d_") or v.endswith("_lag1")
                  or v in ("log_mkt_vol", "ea", "lagged_log_vol")], name)

    base = load_text_panel(h)
    add("baseline", fit(base, X5), absorb="ind+year", cluster="firm+year")

    # #30 market volatility over the outcome window
    d = load_text_panel(h, extra=("log_mkt_vol",))
    add("mktvol", fit(d, X5 + ["log_mkt_vol"]), absorb="ind+year", cluster="firm+year")
    # #32 earnings releases: the EA dummy is ~0 at h <= 30d and ~1 at h >= 90d,
    # so (i) drop windows that contain a release and (ii) drop release days
    d = base[pd.to_numeric(base["ea"], errors="coerce") != 1]
    add("excl_ea_windows", fit(d, X5), absorb="ind+year", cluster="firm+year",
        n_ea_windows_dropped=int(len(base) - len(d)))
    d = load_text_panel(h, dv="exea")
    add("vol_excl_ea_days", fit(d, X5), absorb="ind+year", cluster="firm+year", dv="exea")
    # #30 filing-month x year FE
    d = load_text_panel(h, extra=("fm_year",))
    add("fm_year_fe", fit(d, X5, absorb=("ind", "fm_year")), absorb="ind+filing_month_x_year",
        cluster="firm+year")
    # #30 idiosyncratic volatility as DV
    d = load_text_panel(h, dv="idio")
    add("idio_vol", fit(d, X5), absorb="ind+year", cluster="firm+year", dv="idio")
    # #33 count specification
    d = load_text_panel(h, xvars=CONTROLS + COUNT_VARS)
    add("counts", fit(d, CONTROLS + COUNT_VARS), absorb="ind+year", cluster="firm+year")
    # #52 finer industry; regulated industries out
    d = load_text_panel(h, ind="sic3")
    add("sic3_fe", fit(d, X5), absorb="sic3+year", cluster="firm+year")
    d = load_text_panel(h, ind="sic2")
    add("sic2_snapshot_fe", fit(d, X5), absorb="sic2_today+year", cluster="firm+year")
    d = load_text_panel(h, ind="ff48")
    add("ff48_fe", fit(d, X5), absorb="ff48+year", cluster="firm+year")
    # market cap from weighted-average basic shares (multi-class filers)
    fb = base["shares_source"].astype(str).str.contains("WeightedAverage")
    add("drop_shares_fallback", fit(base[~fb], X5), absorb="ind+year",
        cluster="firm+year", n_dropped=int(fb.sum()))
    hist = base["sic_hist"] if "sic_hist" in base else base["sic"]
    d = base[~(hist.between(4900, 4999) | hist.between(8000, 8099))]
    add("excl_sic49_80", fit(d, X5), absorb="ind+year", cluster="firm+year")
    # #60 post-2019 interaction (main effect absorbed by year FE)
    d = base.copy()
    for v in DECOMP:
        d[f"{v}_x_post"] = d[v] * d["post2019"]
    add("post2019_interaction", fit(d, X5 + [f"{v}_x_post" for v in DECOMP]),
        absorb="ind+year", cluster="firm+year")
    # #47 timing: lagged text; level(t-1) + change
    d = _timing_panel(h)
    lag = [f"{v}_lag1" for v in DECOMP]
    dd = d.dropna(subset=lag)
    add("text_lag1", fit(dd, CONTROLS + lag), absorb="ind+year", cluster="firm+year")
    dch = [f"d_{v}" for v in DECOMP]
    dd = d.dropna(subset=lag + dch)
    add("text_lag1_plus_change", fit(dd, CONTROLS + lag + dch), absorb="ind+year",
        cluster="firm+year")
    # #48 balanced sub-panel: firms present in every FY 2012-2024
    d = base[base["fiscal_year"].between(2012, 2024)]
    full = d.groupby("ticker")["fiscal_year"].nunique() == 13
    d = d[d["ticker"].isin(full[full].index)]
    add("balanced_2012_2024", fit(d, X5), absorb="ind+year", cluster="firm+year")
    # #6 minimum years per firm in the estimation sample
    ny = base.groupby("ticker")["fiscal_year"].transform("nunique")
    for k in (3, 5, 7):
        add(f"min_years_{k}", fit(base[ny >= k], X5), absorb="ind+year",
            cluster="firm+year")
    # #17 drop imputed rows
    flags = [c for c in base.columns if c.startswith("imputed_")]
    imp = base[flags].apply(_bool).any(axis=1) if flags else pd.Series(False, index=base.index)
    add("drop_imputed", fit(base[~imp], X5), absorb="ind+year", cluster="firm+year",
        n_imputed_dropped=int(imp.sum()))
    # lagged DV as an optional extra
    d = load_text_panel(h, extra=("lagged_log_vol",))
    add("plus_lagged_dv", fit(d, X5 + ["lagged_log_vol"]), absorb="ind+year",
        cluster="firm+year")
    # #8 one-way clustering alternatives
    add("cluster_firm", fit(base, X5, cluster=("ticker",)), absorb="ind+year", cluster="firm")
    add("cluster_year", fit(base, X5, cluster=("fiscal_year",)), absorb="ind+year",
        cluster="year")
    return save(rows, f"robustness_{h}")


def wild_bootstrap(h: str = HEADLINE, B: int = 9999, seed: int = 20261101
                   ) -> pd.DataFrame:
    """Wild-cluster bootstrap by fiscal year (Webb weights, null imposed),
    FE partialled out (FWL), for the text coefficients (#8)."""
    import statsmodels.api as sm
    from wildboottest.wildboottest import wildboottest
    rows = []
    base = load_text_panel(h)
    for absorb, an in ((("ind", "fiscal_year"), "ind+year"),
                       (("ticker", "fiscal_year"), "firm+year")):
        for col in ("(4)", "(5)"):
            xs = CONTROLS + SPEC_COLUMNS[col]
            r = fit(base, xs, absorb=absorb)
            mod = AbsorbingLS(r.sample["log_vol"],
                              r.sample[xs].astype(float).assign(const=1.0),
                              absorb=pd.DataFrame({a: r.sample[a].astype(str).astype("category")
                                                   for a in absorb}),
                              drop_absorbed=True)
            mod.fit()
            yw = pd.Series(np.asarray(mod.absorbed_dependent).ravel(), name="y")
            Xw = pd.DataFrame(np.asarray(mod.absorbed_exog), columns=mod.exog.cols)
            Xw = Xw.drop(columns=[c for c in Xw.columns if c == "const"])
            ols = sm.OLS(yw.values, Xw)
            cl = r.sample["fiscal_year"].to_numpy()
            for v in SPEC_COLUMNS[col]:
                out = wildboottest(ols, B=B, cluster=cl, param=v, weights_type="webb",
                                   seed=seed, show=False)
                p_boot = float(np.asarray(out["p-value"]).ravel()[0])
                rows.append({"model": "wild_bootstrap", "column": f"{col}_{an}", "var": v,
                             "coef": float(r.params[v]), "t_twoway": float(r.tstats[v]),
                             "p_twoway": float(r.pvalues[v]), "p_wild_year": p_boot,
                             "B": B, "weights": "webb", "clusters": int(len(np.unique(cl))),
                             "n": int(r.nobs), "h": h, "absorb": an})
                print(f"  wild {an} {col} {v}: p2way={r.pvalues[v]:.4f}  pboot={p_boot:.4f}")
    return save(rows, f"wild_bootstrap_{h}")


def effect_sizes(h: str = HEADLINE) -> pd.DataFrame:
    """Effect of +1 pooled SD and +1 within-firm SD of each text variable,
    in % volatility (#54), under industry FE and firm FE."""
    base = load_text_panel(h)
    rows = []
    for an, absorb in (("ind+year", ("ind", "fiscal_year")),
                       ("firm+year", ("ticker", "fiscal_year"))):
        for col in ("(4)", "(5)"):
            r = fit(base, CONTROLS + SPEC_COLUMNS[col], absorb=absorb)
            for v in SPEC_COLUMNS[col]:
                x = base[v]
                sd_p = float(x.std())
                sd_w = float((x - x.groupby(base["ticker"]).transform("mean")).std())
                b = float(r.params[v])
                rows.append({"var": v, "column": col, "absorb": an, "h": h, "coef": b,
                             "t": float(r.tstats[v]), "p": float(r.pvalues[v]),
                             "sd_pooled": sd_p, "sd_within": sd_w,
                             "pct_per_sd_pooled": 100 * (np.exp(b * sd_p) - 1),
                             "pct_per_sd_within": 100 * (np.exp(b * sd_w) - 1),
                             "n": int(r.nobs)})
    return save(rows, f"effect_sizes_{h}")


def variance_decomposition(h: str = HEADLINE) -> pd.DataFrame:
    """Within-firm / between-firm variance shares of each variable (#62)."""
    base = load_text_panel(h)
    cols = ["log_vol", *TEXT_VARS]
    if "omission_1a" in base:
        cols.append("omission_1a")
    rows = []
    for v in cols:
        x = base[v].dropna()
        g = base.loc[x.index, "ticker"]
        within = x - x.groupby(g).transform("mean")
        tot = float(((x - x.mean()) ** 2).sum())
        w = float((within ** 2).sum())
        yr = x - x.groupby(base.loc[x.index, "fiscal_year"]).transform("mean")
        w_yr = yr - yr.groupby(g).transform("mean")
        rows.append({"var": v, "n": int(len(x)), "sd_total": float(x.std()),
                     "sd_within": float(within.std()), "share_within": w / tot,
                     "share_between": 1 - w / tot,
                     "share_within_net_of_year": float((w_yr ** 2).sum()) / tot})
    return save(rows, f"variance_decomposition_{h}")


def descriptives(h: str = HEADLINE) -> pd.DataFrame:
    base = load_text_panel(h)
    base["vol"] = np.exp(base["log_vol"])
    cols = ["vol", "log_vol", *CONTROLS, "words_1a", *TEXT_VARS, "log_n_unc_1a",
            "log_n_lit_1a"]
    q = base[cols].describe(percentiles=[.01, .25, .5, .75, .99]).T
    q.index.name = "var"
    q.to_csv(RESULTS_DIR / f"summary_stats_{h}.csv")
    base[["log_vol", *CONTROLS, *TEXT_VARS]].corr().to_csv(
        RESULTS_DIR / f"correlation_{h}.csv")
    print(f"  -> summary_stats_{h}.csv, correlation_{h}.csv  "
          f"(N={len(base):,}, firms={base['ticker'].nunique()}, "
          f"median words={base['words_1a'].median():,.0f}, "
          f"max RiskDensity={base[DENS_VAR].max():.4f})")
    return q


ANALYSES = {
    "describe": lambda a: descriptives(),
    "main": lambda a: (run_main("30d"), run_main("365d")),
    "horizons": lambda a: run_horizons(),
    "firm_fe": lambda a: run_firm_fe(),
    "robustness": lambda a: (run_robustness("30d"), run_robustness("365d")),
    "bootstrap": lambda a: wild_bootstrap(B=a.boot),
    "effects": lambda a: effect_sizes(),
    "variance": lambda a: variance_decomposition(),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=list(ANALYSES))
    ap.add_argument("--boot", type=int, default=9999)
    a = ap.parse_args()
    for k in a.only:
        print(f"\n=== {k} ===")
        ANALYSES[k](a)


if __name__ == "__main__":
    main()

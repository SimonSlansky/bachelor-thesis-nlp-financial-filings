"""H3 robustness (results store): firm+year FE (and SIC2+year FE), h = 30d.

Columns (each the H3 specification with one change):
  baseline        omission_1a
  fixed_length    omission_fixedlen_1a (#38, #64)
  length_resid    omission residual on ln Words1A within (SIC2, year) (#64)
  decile_rank     decile rank within (SIC2, year), cells with >= 10 firms (#65)
  tail_deciles    top / bottom decile dummies (#65)
  peers_ge20      cells with >= 20 peers (#66)
  extra_length    + ln Words MD&A + (ln Words1A)^2
  prior_year      omission of fiscal year t-1 (#67: timing / reverse causality)
  pooled_vocab    omission_pooled_1a (#37)
  sic_today       omission with today's SIC2 peers (#72)
  plus_lagged_dv  + lagged log volatility
Also: r(omission, ln Words1A) on the H3 sample (#38), the decile N drop (#65)
and the industry composition of the >= 20-peer sample (#66).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regressions import (CONTROLS, LEN_VAR, DENS_VAR, RESULTS_DIR,  # noqa: E402
                         load_text_panel, fit, save, _read_panel)
from panel import consecutive_lag  # noqa: E402

H = "30d"
OM = "omission_1a"
FE = {"firm+year": ("ticker", "fiscal_year"), "ind+year": ("ind", "fiscal_year")}


def _within_cell_resid(d: pd.DataFrame) -> pd.Series:
    def one(g):
        if len(g) < 5 or g[LEN_VAR].std() == 0:
            return pd.Series(np.nan, index=g.index)
        b = np.polyfit(g[LEN_VAR], g[OM], 1)
        return g[OM] - np.polyval(b, g[LEN_VAR])
    return d.groupby(["ind", "fiscal_year"], group_keys=False)[[LEN_VAR, OM]].apply(one)


def _deciles(d: pd.DataFrame) -> pd.Series:
    return d.groupby(["ind", "fiscal_year"])[OM].transform(
        lambda s: pd.qcut(s, 10, labels=False, duplicates="drop") if s.nunique() >= 10
        else pd.Series(np.nan, index=s.index))


def main():
    base = load_text_panel(H, xvars=CONTROLS + [LEN_VAR, DENS_VAR, OM])
    X = CONTROLS + [LEN_VAR, DENS_VAR]
    base["om_resid"] = _within_cell_resid(base)
    base["om_dec"] = _deciles(base)
    base["om_top"] = (base["om_dec"] == base.groupby(["ind", "fiscal_year"])["om_dec"]
                      .transform("max")).astype(float).where(base["om_dec"].notna())
    base["om_bottom"] = (base["om_dec"] == 0).astype(float).where(base["om_dec"].notna())
    base["log_words_1a_sq"] = base[LEN_VAR] ** 2
    full = _read_panel()[["ticker", "fiscal_year", OM]].sort_values(["ticker", "fiscal_year"])
    full["om_lag1"] = consecutive_lag(full.reset_index(drop=True), OM).values
    base = base.merge(full[["ticker", "fiscal_year", "om_lag1"]], on=["ticker", "fiscal_year"],
                      how="left")

    specs = {
        "baseline": (OM, [], None),
        "fixed_length": ("omission_fixedlen_1a", [], None),
        "length_resid": ("om_resid", [], None),
        "decile_rank": ("om_dec", [], None),
        "tail_deciles": ("om_top", ["om_bottom"], None),
        "peers_ge20": (OM, [], lambda d: d[d["peer_n_1a"] >= 20]),
        "extra_length": (OM, ["log_words_mda", "log_words_1a_sq"], None),
        "prior_year": ("om_lag1", [], None),
        "pooled_vocab": ("omission_pooled_1a", [], None),
        "sic_today": ("omission_1a_sicnow", [], None),
        "plus_lagged_dv": (OM, ["lagged_log_vol"], None),
    }
    rows = []
    for an, absorb in FE.items():
        for name, (var, extra, restrict) in specs.items():
            d = base.dropna(subset=[var, *extra])
            if restrict is not None:
                d = restrict(d)
            r = fit(d, X + [var] + extra, absorb=absorb)
            for v in [var] + extra:
                rows.append({"model": "h3_robust", "column": name, "absorb": an, "var": v,
                             "coef": float(r.params[v]), "se": float(r.std_errors[v]),
                             "t": float(r.tstats[v]), "p": float(r.pvalues[v]),
                             "n": int(r.nobs), "firms": r.n_firms, "h": H})
            print(f"  {an:9s} {name:15s} {var:22s} b={r.params[var]:+.3f} "
                  f"t={r.tstats[var]:+.2f} n={int(r.nobs)}")
    save(rows, "h3_robust")

    # diagnostics
    diag = []
    for v in (OM, "omission_fixedlen_1a", "omission_pooled_1a", "om_resid"):
        d = base[[v, LEN_VAR]].dropna()
        diag.append({"stat": f"corr_{v}_log_words_1a", "value": d.corr().iloc[0, 1],
                     "n": len(d), "sample": "H3 regression sample, h=30d"})
    raw = _read_panel()[[OM, LEN_VAR, "words_1a"]].dropna()
    raw = raw[raw["words_1a"] >= 100]
    diag.append({"stat": "corr_omission_1a_log_words_1a_all_valid", "value":
                 raw.corr().iloc[0, 1], "n": len(raw),
                 "sample": "all firm-years with omission and >=100 words"})
    cell_n = base.groupby(["ind", "fiscal_year"])[OM].transform("nunique")
    diag += [{"stat": "decile_n_dropped_cells_lt10", "value": int((cell_n < 10).sum()),
              "n": len(base), "sample": "H3 sample"},
             {"stat": "decile_n_kept", "value": int(base["om_dec"].notna().sum()),
              "n": len(base), "sample": "H3 sample"},
             {"stat": "fixedlen_short_share", "value":
              float(pd.to_numeric(base.get("fixedlen_short_1a"), errors="coerce").mean()),
              "n": len(base), "sample": "H3 sample"}]
    pd.DataFrame(diag).to_csv(RESULTS_DIR / "h3_diagnostics.csv", index=False)

    big = base[base["peer_n_1a"] >= 20]
    comp = (big.groupby("ind").agg(firm_years=("ticker", "size"), firms=("ticker", "nunique"))
            .assign(share=lambda x: x.firm_years / x.firm_years.sum())
            .sort_values("firm_years", ascending=False))
    comp["share_full_h3"] = base.groupby("ind").size().reindex(comp.index) / len(base)
    comp.to_csv(RESULTS_DIR / "h3_peers_ge20_composition.csv")
    print(pd.DataFrame(diag).to_string(index=False))
    print(comp.head(10))


if __name__ == "__main__":
    main()

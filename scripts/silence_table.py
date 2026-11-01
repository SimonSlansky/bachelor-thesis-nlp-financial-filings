"""H3 main results: omission gap and post-filing volatility (results store).

Grid: h in {5,10,30,90,180} (Holm family) plus 365 (reported outside the
family) x {SIC2+year FE, firm+year FE}, for the headline measure and the
length-invariant measure.  Common controls and loader from regressions.py.

Writes data/results/h3_main.csv and data/results/h3_verdict.json; the
verdict follows the rule recorded in docs/revision_log.md (#42).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from regressions import (CONTROLS, LEN_VAR, DENS_VAR, RESULTS_DIR,  # noqa: E402
                         load_text_panel, fit, save)

FAMILY = ["5d", "10d", "30d", "90d", "180d"]
OUTSIDE = ["365d"]
FE = {"ind+year": ("ind", "fiscal_year"), "firm+year": ("ticker", "fiscal_year")}
MEASURES = {"omission_1a": "headline", "omission_fixedlen_1a": "fixed_length"}
ALPHA = 0.05


def h3_xvars(om: str = "omission_1a") -> list[str]:
    return CONTROLS + [LEN_VAR, DENS_VAR, om]


def h3_sample(h: str, om: str = "omission_1a", **kw) -> pd.DataFrame:
    return load_text_panel(h, xvars=h3_xvars(om), **kw)


def holm(p: pd.Series) -> pd.Series:
    order = p.sort_values().index
    m = len(p)
    adj, run = {}, 0.0
    for k, ix in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[ix]))
        adj[ix] = run
    return pd.Series(adj).reindex(p.index)


def estimate() -> pd.DataFrame:
    rows = []
    for om, label in MEASURES.items():
        for h in FAMILY + OUTSIDE:
            df = h3_sample(h, om)
            for an, absorb in FE.items():
                r = fit(df, h3_xvars(om), absorb=absorb)
                rows.append({"measure": om, "label": label, "h": h, "absorb": an,
                             "in_family": h in FAMILY, "coef": float(r.params[om]),
                             "se": float(r.std_errors[om]), "t": float(r.tstats[om]),
                             "p": float(r.pvalues[om]), "n": int(r.nobs),
                             "firms": r.n_firms, "r2_within": r.r2_within,
                             "coef_len": float(r.params[LEN_VAR]),
                             "t_len": float(r.tstats[LEN_VAR]),
                             "coef_dens": float(r.params[DENS_VAR]),
                             "t_dens": float(r.tstats[DENS_VAR]),
                             "sd_within": float((df[om] - df.groupby("ticker")[om]
                                                 .transform("mean")).std()),
                             "sd_pooled": float(df[om].std())})
                print(f"  {om:22s} {h:>4s} {an:9s} b={r.params[om]:+.3f} "
                      f"t={r.tstats[om]:+.2f} p={r.pvalues[om]:.4f} n={int(r.nobs)}")
    res = pd.DataFrame(rows)
    res["p_holm"] = np.nan
    for om in MEASURES:
        fam = (res["measure"] == om) & res["in_family"]
        res.loc[fam, "p_holm"] = holm(res.loc[fam, "p"]).values
    return res


def verdict(res: pd.DataFrame) -> dict:
    ev = {}
    for an in FE:
        hl = res[(res.measure == "omission_1a") & (res.h == "30d") & (res.absorb == an)].iloc[0]
        fl = res[(res.measure == "omission_fixedlen_1a") & (res.h == "30d")
                 & (res.absorb == an)].iloc[0]
        pos = hl.coef > 0 and hl.p_holm < ALPHA and fl.coef > 0 and fl.p < ALPHA
        neg = hl.coef < 0 and hl.p_holm < ALPHA and fl.coef < 0 and fl.p < ALPHA
        ev[an] = {"positive": bool(pos), "negative": bool(neg),
                  "coef": float(hl.coef), "p": float(hl.p), "p_holm": float(hl.p_holm),
                  "coef_fixedlen": float(fl.coef), "p_fixedlen": float(fl.p)}
    npos = sum(e["positive"] for e in ev.values())
    nneg = sum(e["negative"] for e in ev.values())
    if npos == 2:
        v = "supported"
    elif npos == 1 and nneg == 0:
        v = "partially supported"
    elif nneg >= 1 and npos == 0:
        v = "contradicted"
    else:
        v = "not supported"
    return {"verdict": v, "rule": "docs/revision_log.md, H3 decision rule (#42)",
            "evidence": ev}


def main():
    res = estimate()
    save(res.to_dict("records"), "h3_main")
    v = verdict(res)
    (RESULTS_DIR / "h3_verdict.json").write_text(json.dumps(v, indent=2))
    print(json.dumps(v, indent=2))


if __name__ == "__main__":
    main()

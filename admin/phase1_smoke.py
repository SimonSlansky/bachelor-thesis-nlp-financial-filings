"""Phase-1 smoke test (no table output): specs (a)-(e) at h = 30 days.

(a) original model: lagged ln vol + size/leverage/ROA/asset growth, today's
    SIC2, no liquidity / text-mismatch rules
(b) common control vector (today's SIC2)
(c) (b) + ln market vol + earnings-announcement dummy
(d) (b) with ln idiosyncratic vol as DV
(e) (b) with historical SIC2
Each: decomposition spec (col 5) and composite spec (col 4).
Writes admin/phase1_smoke.csv and prints the comparison.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from regressions import (CONTROLS, ORIG_CONTROLS, DECOMP, JOINT,  # noqa: E402
                         load_text_panel, fit)

SPECS = {
    "(a) original": dict(controls=ORIG_CONTROLS, ind="sic2", liquid=False,
                         drop_mismatch=False),
    "(b) common controls": dict(controls=CONTROLS, ind="sic2"),
    "(c) (b)+mkt vol+EA": dict(controls=CONTROLS + ["log_mkt_vol", "ea"], ind="sic2"),
    "(d) (b) idio vol DV": dict(controls=CONTROLS, ind="sic2", dv="idio"),
    "(e) (b) hist SIC2": dict(controls=CONTROLS, ind="sic2_hist"),
}
SHOW = ["log_words_1a", "unc_1a", "lit_1a", "risk_1a", "pre_log_vol",
        "lagged_log_vol", "log_mkt_vol", "ea"]


def main():
    rows = []
    for name, s in SPECS.items():
        ctrl = s.pop("controls")
        df = load_text_panel("30d", xvars=ctrl + DECOMP + ["risk_1a"], **s)
        for col, text in (("col5", DECOMP), ("col4", JOINT)):
            r = fit(df, ctrl + text)
            for v in SHOW:
                if v in r.params.index:
                    rows.append({"spec": name, "column": col, "var": v,
                                 "coef": r.params[v], "t": r.tstats[v], "p": r.pvalues[v],
                                 "n": int(r.nobs), "firms": r.n_firms,
                                 "r2_within": r.r2_within})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "admin" / "phase1_smoke.csv", index=False)
    out["cell"] = out.apply(lambda x: f"{x.coef:+.3f} ({x.t:+.2f})", axis=1)
    wide = out.pivot_table(index=["column", "var"], columns="spec", values="cell",
                           aggfunc="first")
    pd.set_option("display.width", 250)
    print(wide.to_string())
    print(out.groupby("spec")[["n", "firms"]].first())


if __name__ == "__main__":
    main()

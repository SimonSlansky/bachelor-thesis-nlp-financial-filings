"""
Generate tex/tables/silence_main.tex — main results table for the
Silence Hypothesis (Item 1A omission relative to SIC2-year peers).

Columns: horizons 5, 10, 30, 90, 180 days
Panels:
  A. Industry + year FE  (cross-section)
  B. Firm + year FE      (within-firm — the headline test)

Each cell reports β and t-statistic (two-way clustered on ticker, year)
for omission_1a, on top of: log(lagged_vol_h), log_total_assets, leverage,
roa, current_ratio, ocf_to_assets, log_words_1a, risk_1a.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from linearmodels.iv.absorbing import AbsorbingLS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

HORIZONS = [5, 10, 30, 90, 180]
FIN = ["log_total_assets", "leverage", "roa", "current_ratio", "ocf_to_assets"]


def fit(df, h, firm_fe):
    cols = (["ticker", "fiscal_year", "sic", f"vol_{h}d", f"lagged_vol_{h}d",
             "log_words_1a", "risk_1a", "omission_1a"] + FIN)
    d = df[cols].dropna().copy()
    d = d[(d[f"vol_{h}d"] > 0) & (d[f"lagged_vol_{h}d"] > 0)]
    if len(d) < 200:
        return None
    d["sic2"] = (d["sic"] // 100).astype("category")
    d["fy"] = d["fiscal_year"].astype("category")
    d["tk"] = d["ticker"].astype("category")
    X = d[[f"lagged_vol_{h}d", *FIN, "log_words_1a", "risk_1a",
           "omission_1a"]].copy()
    X[f"lagged_vol_{h}d"] = np.log(X[f"lagged_vol_{h}d"])
    X.insert(0, "const", 1.0)
    y = np.log(d[f"vol_{h}d"])
    absorb = d[["tk", "fy"]] if firm_fe else d[["sic2", "fy"]]
    clusters = d[["tk", "fy"]].apply(lambda c: c.cat.codes)
    m = AbsorbingLS(y, X, absorb=absorb, drop_absorbed=True)
    r = m.fit(cov_type="clustered", clusters=clusters)
    return {"n": int(r.nobs), "r2": float(r.rsquared),
            "b": float(r.params["omission_1a"]),
            "se": float(r.std_errors["omission_1a"]),
            "t": float(r.tstats["omission_1a"]),
            "p": float(r.pvalues["omission_1a"])}


def stars(p):
    return ("***" if p < 0.01 else
            "**" if p < 0.05 else
            "*" if p < 0.1 else "")


def main():
    panel = pd.read_csv(DATA / "annual_panel_text.csv")
    sil = pd.read_csv(DATA / "silence.csv")
    df = panel.merge(sil, on=["ticker", "fiscal_year"], how="left")

    rows = {"ind": {}, "firm": {}}
    for h in HORIZONS:
        rows["ind"][h] = fit(df, h, False)
        rows["firm"][h] = fit(df, h, True)

    def cell(r):
        if r is None: return "--"
        return f"{r['b']:+.3f}{stars(r['p'])}"
    def cell_t(r):
        if r is None: return ""
        return f"({r['t']:+.2f})"

    hh = " & ".join(f"$h={h}$" for h in HORIZONS)
    ind_b = " & ".join(cell(rows["ind"][h]) for h in HORIZONS)
    ind_t = " & ".join(cell_t(rows["ind"][h]) for h in HORIZONS)
    fir_b = " & ".join(cell(rows["firm"][h]) for h in HORIZONS)
    fir_t = " & ".join(cell_t(rows["firm"][h]) for h in HORIZONS)
    ind_n = " & ".join(f"{rows['ind'][h]['n']:,}" for h in HORIZONS)
    fir_n = " & ".join(f"{rows['firm'][h]['n']:,}" for h in HORIZONS)
    ind_r = " & ".join(f"{rows['ind'][h]['r2']:.3f}" for h in HORIZONS)
    fir_r = " & ".join(f"{rows['firm'][h]['r2']:.3f}" for h in HORIZONS)

    tex = (
        r"\begin{table}[htbp]" "\n"
        r"\centering" "\n"
        r"\caption{Silence Hypothesis: Item 1A omission relative to SIC2 peers and post-filing realised volatility.}" "\n"
        r"\label{tab:silence_main}" "\n"
        r"\footnotesize" "\n"
        r"\begin{tabular}{lccccc}" "\n"
        r"\toprule" "\n"
        f"& {hh} \\\\\n"
        r"\midrule" "\n"
        r"\multicolumn{6}{l}{\textit{Panel A. Industry $\times$ year fixed effects}} \\" "\n"
        f"$\\textsc{{Omission}}_{{1A}}$ & {ind_b} \\\\\n"
        f"& {ind_t} \\\\\n"
        f"$N$ & {ind_n} \\\\\n"
        f"$R^2$ & {ind_r} \\\\\n"
        r"\midrule" "\n"
        r"\multicolumn{6}{l}{\textit{Panel B. Firm $\times$ year fixed effects}} \\" "\n"
        f"$\\textsc{{Omission}}_{{1A}}$ & {fir_b} \\\\\n"
        f"& {fir_t} \\\\\n"
        f"$N$ & {fir_n} \\\\\n"
        f"$R^2$ & {fir_r} \\\\\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\par\smallskip" "\n"
        r"\begin{flushleft}\footnotesize" "\n"
        r"\textit{Notes.} Dependent variable is $\log(\textrm{RV}_{h\textrm{d}})$ over the $h$-day window beginning the trading day after filing. "
        r"$\textsc{Omission}_{1A}$ is the share of TF-IDF mass on terms used by SIC2-year peers (leave-one-out, $\geq 5$ peers) that the focal firm does not mention. "
        r"All regressions control for $\log(\textrm{RV}_{h\textrm{d}, t-1})$, the five financial controls (size, leverage, ROA, current ratio, OCF/assets), $\log(\textrm{words}_{1A})$, and $\textsc{RiskDensity}_{1A}$. "
        r"$t$-statistics in parentheses use two-way clustered standard errors on (ticker, fiscal year). "
        r"$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$." "\n"
        r"\end{flushleft}" "\n"
        r"\end{table}" "\n"
    )
    out_path = ROOT / "tex" / "tables" / "silence_main.tex"
    out_path.write_text(tex, encoding="utf-8")
    print(f"Wrote {out_path}")
    print()
    print("Headline numbers:")
    for h in HORIZONS:
        r = rows["firm"][h]
        print(f"  firm+year FE, h={h:3d}d  β={r['b']:+.3f}  "
              f"t={r['t']:+5.2f}  p={r['p']:.3f}")


if __name__ == "__main__":
    main()

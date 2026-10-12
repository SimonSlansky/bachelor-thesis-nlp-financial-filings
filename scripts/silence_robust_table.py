"""
Generate tex/tables/silence_robust.tex — robustness panel for the
H4 strategic-silence specification at h=30d under firm+year FE.

Six columns:
  (1) baseline:   omission_1a + log_words_1a + RiskDensity + 5 fin + lagged
  (2) decile:     omission_1a replaced by within-(sic2,year) decile rank
  (3) length-resid: replace omission_1a with residual after regressing on log_words_1a within (sic2,year)
  (4) large cells: restrict to peer_n_1a >= 20
  (5) triple len:  add log_words_mda + log_words_1a^2 controls
  (6) lag-1:       use omission_1a_lag1 instead of omission_1a

Also report placebo results in a footnote / textual note.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from linearmodels.iv.absorbing import AbsorbingLS

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
H = 30
FIN = ["log_total_assets", "leverage", "roa", "current_ratio", "ocf_to_assets"]


def fit(df, var, extra_controls=None, restrict=None):
    extra_controls = extra_controls or []
    cols = (["ticker", "fiscal_year", "sic", f"vol_{H}d", f"lagged_vol_{H}d",
             "log_words_1a", "risk_1a", "peer_n_1a", var] + FIN + extra_controls)
    cols = list(dict.fromkeys(cols))
    cols = [c for c in cols if c in df.columns]
    d = df[cols].dropna(subset=[c for c in cols if c != "peer_n_1a"]).copy()
    d = d[(d[f"vol_{H}d"] > 0) & (d[f"lagged_vol_{H}d"] > 0)]
    if restrict is not None:
        d = restrict(d)
    if len(d) < 200:
        return None
    d["fy"] = d["fiscal_year"].astype("category")
    d["tk"] = d["ticker"].astype("category")
    X = d[[f"lagged_vol_{H}d", *FIN, "log_words_1a", "risk_1a", var,
           *extra_controls]].copy()
    X[f"lagged_vol_{H}d"] = np.log(X[f"lagged_vol_{H}d"])
    X.insert(0, "const", 1.0)
    y = np.log(d[f"vol_{H}d"])
    absorb = d[["tk", "fy"]]
    clusters = absorb.apply(lambda c: c.cat.codes)
    m = AbsorbingLS(y, X, absorb=absorb, drop_absorbed=True)
    r = m.fit(cov_type="clustered", clusters=clusters)
    return {"n": int(r.nobs),
            "b": float(r.params.get(var, np.nan)),
            "se": float(r.std_errors.get(var, np.nan)),
            "t": float(r.tstats.get(var, np.nan)),
            "p": float(r.pvalues.get(var, np.nan))}


def stars(p):
    return ("***" if p < 0.01 else
            "**" if p < 0.05 else
            "*" if p < 0.1 else "")


def main():
    panel = pd.read_csv(DATA / "annual_panel_text.csv")
    sil = pd.read_csv(DATA / "silence.csv")
    df = panel.merge(sil, on=["ticker", "fiscal_year"], how="left")

    # Length-residualised omission (within sic2, year)
    df["sic2"] = (df["sic"] // 100).astype("Int64")
    valid = df.dropna(subset=["omission_1a", "log_words_1a", "sic", "fiscal_year"]).copy()
    def resid(g):
        if len(g) < 5 or g["log_words_1a"].std() == 0:
            return pd.Series(np.nan, index=g.index)
        x = g["log_words_1a"].values
        y = g["omission_1a"].values
        b = np.cov(x, y)[0, 1] / np.var(x)
        a = y.mean() - b * x.mean()
        return pd.Series(y - (a + b * x), index=g.index)
    valid["om_resid"] = (valid.groupby(["sic2", "fiscal_year"], group_keys=False)
                              .apply(resid))
    df = df.merge(valid[["ticker", "fiscal_year", "om_resid"]],
                   on=["ticker", "fiscal_year"], how="left")

    # Decile rank within sic2, year
    valid2 = df.dropna(subset=["omission_1a", "sic2", "fiscal_year"]).copy()
    valid2["om_dec"] = (
        valid2.groupby(["sic2", "fiscal_year"])["omission_1a"]
              .transform(lambda s: pd.qcut(s, 10, labels=False, duplicates="drop")
                         if s.nunique() >= 10 else np.nan)
    )
    df = df.merge(valid2[["ticker", "fiscal_year", "om_dec"]],
                   on=["ticker", "fiscal_year"], how="left")

    # Squared length
    df["log_words_1a_sq"] = df["log_words_1a"] ** 2
    # Lag
    df = df.sort_values(["ticker", "fiscal_year"]).copy()
    df["om_lag1"] = df.groupby("ticker")["omission_1a"].shift(1)

    cols = []
    cols.append(("(1) Baseline", fit(df, "omission_1a")))
    cols.append(("(2) Decile rank", fit(df, "om_dec")))
    cols.append(("(3) Length-resid", fit(df, "om_resid")))
    cols.append(("(4) $\\geq 20$ peers", fit(df, "omission_1a",
                                              restrict=lambda d: d[d["peer_n_1a"] >= 20])))
    cols.append(("(5) Triple length",
                  fit(df, "omission_1a",
                      extra_controls=["log_words_mda", "log_words_1a_sq"])))
    cols.append(("(6) Prior-year",   fit(df, "om_lag1")))

    headers = " & ".join(c[0] for c in cols)
    bs = " & ".join(f"{c[1]['b']:+.3f}{stars(c[1]['p'])}" for c in cols)
    ts = " & ".join(f"({c[1]['t']:+.2f})" for c in cols)
    ns = " & ".join(f"{c[1]['n']:,}" for c in cols)

    tex = (
        r"\begin{table}[htbp]" "\n"
        r"\centering" "\n"
        r"\caption{Robustness of the strategic-silence result (H4): firm $\times$ year fixed effects, $h = 30$ days.}" "\n"
        r"\label{tab:silence_robust}" "\n"
        r"\footnotesize" "\n"
        r"\begin{tabular}{lcccccc}" "\n"
        r"\toprule" "\n"
        f"& {headers} \\\\\n"
        r"\midrule" "\n"
        f"$\\textsc{{Omission}}_{{1A}}$ & {bs} \\\\\n"
        f"& {ts} \\\\\n"
        f"$N$ & {ns} \\\\\n"
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\par\smallskip" "\n"
        r"\begin{flushleft}\footnotesize" "\n"
        r"\textit{Notes.} Each column is a separate firm $\times$ year fixed-effect regression at $h = 30$ days. "
        r"Column~(1) reports the baseline H4 specification of equation~\ref{eq:silence_model}. "
        r"Column~(2) replaces $\textsc{Omission}_{1A}$ with its decile rank within (SIC2, fiscal year) cells. "
        r"Column~(3) replaces it with the residual after regressing $\textsc{Omission}_{1A}$ on $\ln(\textsc{Words}_{1A})$ within (SIC2, fiscal year) cells. "
        r"Column~(4) restricts the sample to industry-year cells with at least 20 panel firms. "
        r"Column~(5) adds $\ln(\textsc{Words}_{\mathrm{MDA}})$ and $\ln(\textsc{Words}_{1A})^2$ as additional length controls. "
        r"Column~(6) replaces the contemporaneous omission with its prior-year value. "
        r"All regressions include lagged log-volatility, the five financial controls, $\ln(\textsc{Words}_{1A})$, and $\textsc{RiskDensity}_{1A}$. "
        r"$t$-statistics in parentheses use two-way clustered standard errors on (ticker, fiscal year). "
        r"$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$." "\n"
        r"\end{flushleft}" "\n"
        r"\end{table}" "\n"
    )
    out = ROOT / "tex" / "tables" / "silence_robust.tex"
    out.write_text(tex, encoding="utf-8")
    print(f"Wrote {out}")
    for label, r in cols:
        print(f"  {label:24s}  β={r['b']:+.4f}  t={r['t']:+5.2f}  "
              f"p={r['p']:.4f}  n={r['n']}")


if __name__ == "__main__":
    main()

"""Write every LaTeX table of the thesis from data/results/ and data/*.csv.

No numbers are typed by hand: each table reads a results file produced by
regressions.py, silence_*.py, stationarity.py, validation_*.py or the Phase-1
diagnostics.  Run after those scripts:  python scripts/make_tables.py
(legacy 10-Q pilot tables q4_reliability, text_availability, xbrl_coverage,
annual_vs_quarterly, text_per_company are not produced here; see README).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "data" / "results"
DATA = ROOT / "data"
TEX = ROOT / "tex" / "tables"

LAB = {
    "log_vol": r"$\ln\sigma^{[30]}_{i,t+1}$", "vol": r"$\sigma^{[30]}_{i,t+1}$",
    "pre_log_vol": r"$\ln\sigma^{\mathrm{pre}}_{i,t}$", "log_mcap": r"$\ln(\text{Market cap})$",
    "leverage": "Leverage", "roa": "ROA", "asset_growth": "Asset growth",
    "bm": "Book-to-market", "ret_pre_12m": "Past 12-month return",
    "current_ratio": "Current ratio", "ocf_to_assets": "OCF/assets",
    "words_1a": r"Words$^{1A}$", "log_words_1a": r"$\ln(\text{Words}^{1A})$",
    "risk_1a": "RiskDensity", "unc_1a": "UncDensity", "lit_1a": "LitDensity",
    "log_n_unc_1a": r"$\ln(1+n^{\text{unc}})$", "log_n_lit_1a": r"$\ln(1+n^{\text{lit}})$",
    "lagged_log_vol": r"$\ln\sigma^{[30]}_{i,t}$ (lagged)", "log_mkt_vol": r"$\ln\sigma^{\mathrm{mkt}}$",
    "omission_1a": r"Omission$^{1A}$",
}
CONTROLS = ["pre_log_vol", "log_mcap", "leverage", "roa", "asset_growth", "bm",
            "ret_pre_12m", "current_ratio", "ocf_to_assets"]
TEXT = ["log_words_1a", "risk_1a", "unc_1a", "lit_1a"]


def stars(p):
    return "" if pd.isna(p) else "***" if p < .01 else "**" if p < .05 else "*" if p < .1 else ""


def cell(c, p, d=3):
    return "" if pd.isna(c) else f"${c:+.{d}f}${stars(p)}"


def tcell(t):
    return "" if pd.isna(t) else f"$({t:+.2f})$"


def n_(x):
    return f"{int(x):,}"


def write(name, lines):
    text = "\n".join(lines) + "\n"
    if "\\cmidrule" in text:
        # babel czech/slovak make "-" active, which breaks \cmidrule{a-b}
        text = text.replace("\\centering", "\\centering\\shorthandoff{-}", 1)
    (TEX / f"{name}.tex").write_text(text, encoding="utf-8")
    print(f"  -> tex/tables/{name}.tex")


def note(text, width="\\linewidth"):
    return [r"\par\smallskip", rf"\begin{{minipage}}{{{width}}}\footnotesize",
            r"\emph{Notes.} " + text, r"\end{minipage}"]


SIG = r"$^{*}p<0.10$, $^{**}p<0.05$, $^{***}p<0.01$."
FE_NOTE = (r"Industry = two-digit SIC code at the filing date. $t$-statistics in parentheses use "
           r"standard errors clustered two-way by firm and fiscal year. ")


# ── descriptive ─────────────────────────────────────────────────────────

def summary():
    s = pd.read_csv(RES / "summary_stats_30d.csv", index_col=0)
    rows = ["vol", "log_vol", *CONTROLS, "words_1a", *TEXT]
    L = [r"\begin{table}[htbp]", r"\centering", r"\caption{Summary statistics (30-day estimation sample)}",
         r"\label{tab:summary}", r"\small", r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r" & Mean & SD & P1 & P25 & Median & P75 & P99 \\", r"\midrule"]
    for v in rows:
        r = s.loc[v]
        fmt = (lambda x: f"{x:,.0f}") if v == "words_1a" else \
              (lambda x: f"{x:.4f}") if v in ("risk_1a", "unc_1a", "lit_1a") else (lambda x: f"{x:.3f}")
        L.append(f"{LAB[v]} & " + " & ".join(fmt(r[k]) for k in
                 ("mean", "std", "1%", "25%", "50%", "75%", "99%")) + r" \\")
    L += [r"\midrule", rf"Observations & \multicolumn{{7}}{{l}}{{{n_(s.loc['log_vol','count'])}}} \\",
          r"\bottomrule", r"\end{tabular}"]
    L += note(r"$\sigma^{[30]}_{i,t+1}$: annualised standard deviation of daily log returns in the "
              r"30-calendar-day window starting two trading days after the 10-K filing day. "
              r"$\sigma^{\mathrm{pre}}$: the same over the 60 trading days before filing. "
              r"Financial ratios, book-to-market and the past return are winsorised at the 1st/99th "
              r"percentiles of this sample; the text variables are not winsorised. "
              rf"The maximum RiskDensity is {s.loc['risk_1a','max']:.4f}.")
    L += [r"\end{table}"]
    write("summary_statistics", L)


def correlation():
    c = pd.read_csv(RES / "correlation_30d.csv", index_col=0)
    vs = ["log_vol", *CONTROLS, *TEXT]
    L = [r"\begin{table}[htbp]", r"\centering", r"\caption{Pearson correlations (30-day estimation sample)}",
         r"\label{tab:correlation}", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}",
         r"\begin{tabular}{l" + "r" * (len(vs) - 1) + "}", r"\toprule",
         " & " + " & ".join(f"({i})" for i in range(1, len(vs))) + r" \\", r"\midrule"]
    for i, v in enumerate(vs):
        cells = [f"{c.loc[v, w]:.2f}" if j < i else "" for j, w in enumerate(vs[:-1])]
        L.append(f"({i + 1}) {LAB[v]} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    L += note(r"$N$ as in Table~\ref{tab:summary}.")
    L += [r"\end{table}"]
    write("correlation_matrix", L)


# ── H1/H2 ───────────────────────────────────────────────────────────────

def main_table(h):
    m = pd.read_csv(RES / f"main_{h}.csv")
    cols = ["(1)", "(2)", "(3)", "(4)", "(5)"]
    vs = TEXT + CONTROLS
    L = [r"\begin{table}[htbp]", r"\centering",
         rf"\caption{{Item~1A length, risk-word density and post-filing volatility ({h[:-1]}-day window)}}",
         rf"\label{{tab:main_{h}}}", r"\small", r"\begin{tabular}{lccccc}", r"\toprule",
         " & " + " & ".join(cols) + r" \\", r"\midrule"]
    for v in vs:
        cs, ts = [], []
        for c in cols:
            r = m[(m.column == c) & (m["var"] == v)]
            cs.append(cell(r.coef.iloc[0], r.p.iloc[0]) if len(r) else "")
            ts.append(tcell(r.t.iloc[0]) if len(r) else "")
        L += [f"{LAB[v]} & " + " & ".join(cs) + r" \\", " & " + " & ".join(ts) + r" \\[2pt]"]
    first = m[m["var"] == "pre_log_vol"].set_index("column")
    w = m[m["var"] == "wald_unc_eq_lit"].iloc[0]
    L += [r"\midrule", "Industry and year FE & " + " & ".join(["Yes"] * 5) + r" \\",
          "Observations & " + " & ".join(n_(first.loc[c, "n"]) for c in cols) + r" \\",
          "Firms & " + " & ".join(str(int(first.loc[c, "firms"])) for c in cols) + r" \\",
          r"Within $R^2$ & " + " & ".join(f"{first.loc[c, 'r2_within']:.3f}" for c in cols) + r" \\",
          r"$\Delta$ within $R^2$ vs (1) & " + " & ".join(
              f"{first.loc[c, 'incr_r2_within']:.4f}" for c in cols) + r" \\",
          r"Wald $\beta^{\text{unc}}=\beta^{\text{lit}}$, $p$ & & & & & " + f"{w.p:.3f}" + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    L += note(rf"Dependent variable $\ln\sigma^{{[{h[:-1]}]}}_{{i,t+1}}$. All columns use the same "
              r"sample and the common control vector. " + FE_NOTE + r"Within $R^2$: share of the "
              r"variance left after the fixed effects that the regressors explain. " + SIG)
    L += [r"\end{table}"]
    write(f"main_{h}", L)


def horizons():
    m = pd.read_csv(RES / "horizons.csv")
    hs = ["5d", "10d", "30d", "90d", "180d", "365d"]
    L = [r"\begin{table}[htbp]", r"\centering", r"\caption{Text coefficients at alternative horizons}",
         r"\label{tab:horizons}", r"\small", r"\begin{tabular}{lcccccc}", r"\toprule",
         " & " + " & ".join(f"{h[:-1]} days" for h in hs) + r" \\", r"\midrule",
         r"\multicolumn{7}{l}{\emph{Decomposition specification (column 5)}} \\"]
    for v in ["log_words_1a", "unc_1a", "lit_1a"]:
        rr = [m[(m.h == h) & (m.column == "(5)") & (m["var"] == v)].iloc[0] for h in hs]
        L += [f"{LAB[v]} & " + " & ".join(cell(r.coef, r.p) for r in rr) + r" \\",
              " & " + " & ".join(tcell(r.t) for r in rr) + r" \\[2pt]"]
    L.append(r"\multicolumn{7}{l}{\emph{Composite specification (column 4)}} \\")
    for v in ["log_words_1a", "risk_1a"]:
        rr = [m[(m.h == h) & (m.column == "(4)") & (m["var"] == v)].iloc[0] for h in hs]
        L += [f"{LAB[v]} & " + " & ".join(cell(r.coef, r.p) for r in rr) + r" \\",
              " & " + " & ".join(tcell(r.t) for r in rr) + r" \\[2pt]"]
    rr = [m[(m.h == h) & (m.column == "(5)") & (m["var"] == "pre_log_vol")].iloc[0] for h in hs]
    L += [r"\midrule", "Observations & " + " & ".join(n_(r.n) for r in rr) + r" \\",
          r"Within $R^2$ (col.\ 5) & " + " & ".join(f"{r.r2_within:.3f}" for r in rr) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    L += note(r"Each column re-estimates the specification with $\ln\sigma^{[h]}_{i,t+1}$ as the "
              r"dependent variable; common controls, industry and year FE. " + FE_NOTE + SIG)
    L += [r"\end{table}"]
    write("horizons", L)


def firm_fe():
    m = pd.read_csv(RES / "firm_fe.csv")
    spec = [("(5)_no_ldv_ind+year", "Industry + year"), ("(5)_no_ldv_firm+year", "Firm + year"),
            ("(5)_with_ldv_firm+year", "Firm + year, lagged DV")]
    L = [r"\begin{table}[htbp]", r"\centering", r"\caption{Industry versus firm fixed effects (30-day window)}",
         r"\label{tab:firm_fe}", r"\small", r"\begin{tabular}{lccc}", r"\toprule",
         " & " + " & ".join(f"({i + 1}) {s}" for i, (_, s) in enumerate(spec)) + r" \\", r"\midrule"]
    for v in ["log_words_1a", "unc_1a", "lit_1a", "pre_log_vol", "lagged_log_vol"]:
        rr = [m[(m.column == c) & (m["var"] == v)] for c, _ in spec]
        L += [f"{LAB[v]} & " + " & ".join(cell(r.coef.iloc[0], r.p.iloc[0]) if len(r) else "" for r in rr) + r" \\",
              " & " + " & ".join(tcell(r.t.iloc[0]) if len(r) else "" for r in rr) + r" \\[2pt]"]
    rr = [m[(m.column == c.replace("(5)", "(4)")) & (m["var"] == "risk_1a")].iloc[0] for c, _ in spec]
    L += [r"\midrule", r"RiskDensity in column-(4) specification & " +
          " & ".join(cell(r.coef, r.p) for r in rr) + r" \\", " & " + " & ".join(tcell(r.t) for r in rr) + r" \\"]
    rr = [m[(m.column == c) & (m["var"] == "pre_log_vol")].iloc[0] for c, _ in spec]
    L += [r"\midrule", "Observations & " + " & ".join(n_(r.n) for r in rr) + r" \\",
          "Firms & " + " & ".join(str(int(r.firms)) for r in rr) + r" \\",
          r"Within $R^2$ & " + " & ".join(f"{r.r2_within:.3f}" for r in rr) + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    L += note(r"All columns include the common control vector and use the same sample. Column (3) adds "
              r"the lagged dependent variable, which is subject to Nickell bias under firm fixed effects. "
              r"Standard errors clustered two-way by firm and fiscal year. " + SIG)
    L += [r"\end{table}"]
    write("firm_fe", L)


ROB = [("baseline", "Baseline (Table~\\ref{tab:main_30d}, col.\\ 5)"),
       ("mktvol", "+ market volatility in window"),
       ("excl_ea_windows", "Windows with an earnings release dropped"),
       ("vol_excl_ea_days", "Volatility without earnings-release days"),
       ("idio_vol", "Idiosyncratic volatility as DV"),
       ("fm_year_fe", "Filing-month $\\times$ year FE"),
       ("cal_ym_fe", "Calendar year-month of filing FE (instead of fiscal year)"),
       ("dec_fye_only", "Firms with a December fiscal-year end only"),
       ("sic3_fe", "SIC3 + year FE"), ("ff48_fe", "FF48 + year FE"),
       ("sic2_snapshot_fe", "Today's SIC2 + year FE"),
       ("excl_sic49_80", "Excluding utilities (49) and health services (80)"),
       ("drop_shares_fallback", "Excluding weighted-average share counts"),
       ("drop_imputed", "Excluding imputed balance-sheet values"),
       ("balanced_2012_2024", "Balanced panel, FY 2012--2024"),
       ("min_years_3", "Firms with $\\geq 3$ years"), ("min_years_5", "Firms with $\\geq 5$ years"),
       ("min_years_7", "Firms with $\\geq 7$ years"),
       ("plus_lagged_dv", "+ lagged dependent variable"),
       ("cluster_firm", "SE clustered by firm only"), ("cluster_year", "SE clustered by year only")]


def robustness(h):
    m = pd.read_csv(RES / f"robustness_{h}.csv")
    L = [r"\begin{table}[htbp]", r"\centering",
         rf"\caption{{Robustness of the H1/H2 decomposition ({h[:-1]}-day window)}}",
         rf"\label{{tab:robust_{h}}}", r"\footnotesize", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{lcccr}", r"\toprule",
         r" & $\ln(\text{Words}^{1A})$ & UncDensity & LitDensity & $N$ \\", r"\midrule"]
    for k, lab in ROB:
        r = m[m.column == k].set_index("var")
        if r.empty:
            continue
        L.append(f"{lab} & " + " & ".join(
            f"{cell(r.loc[v, 'coef'], r.loc[v, 'p'])} {tcell(r.loc[v, 't'])}" for v in
            ["log_words_1a", "unc_1a", "lit_1a"]) + f" & {n_(r['n'].iloc[0])}" + r" \\")
    r = m[m.column == "counts"].set_index("var")
    if len(r):
        L.append(r"Counts: $\ln(\text{Words})$, $\ln(1+n^{\text{unc}})$, $\ln(1+n^{\text{lit}})$ & " +
                 " & ".join(f"{cell(r.loc[v, 'coef'], r.loc[v, 'p'])} {tcell(r.loc[v, 't'])}"
                            for v in ["log_words_1a", "log_n_unc_1a", "log_n_lit_1a"]) +
                 f" & {n_(r['n'].iloc[0])}" + r" \\")
    for k, lab, suf in [("text_lag1", "Text of year $t-1$ instead of $t$", "_lag1")]:
        r = m[m.column == k].set_index("var")
        if len(r):
            L.append(f"{lab} & " + " & ".join(
                f"{cell(r.loc[v + suf, 'coef'], r.loc[v + suf, 'p'])} {tcell(r.loc[v + suf, 't'])}"
                for v in ["log_words_1a", "unc_1a", "lit_1a"]) + f" & {n_(r['n'].iloc[0])}" + r" \\")
    r = m[m.column == "text_lag1_plus_change"].set_index("var")
    if len(r):
        for pre, lab in (("", "Level $t-1$"), ("d_", "Change $t-1\\to t$")):
            vs = [f"{v}_lag1" if pre == "" else f"d_{v}" for v in ["log_words_1a", "unc_1a", "lit_1a"]]
            L.append(f"Level + change: {lab} & " + " & ".join(
                f"{cell(r.loc[v, 'coef'], r.loc[v, 'p'])} {tcell(r.loc[v, 't'])}" for v in vs) +
                f" & {n_(r['n'].iloc[0])}" + r" \\")
    r = m[m.column == "post2019_interaction"].set_index("var")
    if len(r):
        for suf, lab in (("", "FY $\\leq$ 2019 (main effect)"), ("_x_post", "$\\times$ FY $\\geq$ 2020")):
            L.append(f"Post-2019 interaction: {lab} & " + " & ".join(
                f"{cell(r.loc[v + suf, 'coef'], r.loc[v + suf, 'p'])} {tcell(r.loc[v + suf, 't'])}"
                for v in ["log_words_1a", "unc_1a", "lit_1a"]) + f" & {n_(r['n'].iloc[0])}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    L += note(r"Each row is a separate regression of the column-(5) specification with the change "
              r"described; common controls, industry (historical SIC2) and year FE and two-way clustered "
              r"standard errors unless stated. " + SIG)
    L += [r"\end{table}"]
    write(f"robustness_{h}", L)


def inference():
    b = pd.read_csv(RES / "wild_bootstrap_30d.csv")
    r = pd.read_csv(RES / "robustness_30d.csv")
    L = [r"\begin{table}[htbp]", r"\centering", r"\caption{Inference with few year clusters (30-day window)}",
         r"\label{tab:inference}", r"\small", r"\begin{tabular}{llcccc}", r"\toprule",
         r" & & Coef. & \multicolumn{3}{c}{$p$-value} \\ \cmidrule(lr){4-6}",
         r"FE & Variable & & Two-way & Firm & Wild bootstrap (year) \\", r"\midrule"]
    for _, x in b.iterrows():
        fe = "Industry + year" if "ind" in x.column else "Firm + year"
        pf = ""
        if "ind" in x.column and x.column.startswith("(5)"):
            q = r[(r.column == "cluster_firm") & (r["var"] == x["var"])]
            pf = f"{q.p.iloc[0]:.3f}" if len(q) else ""
        L.append(f"{fe} & {LAB[x['var']]} {x.column[:3]} & ${x.coef:+.3f}$ & {x.p_twoway:.3f} & {pf} & "
                 f"{x.p_wild_year:.3f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    L += note(rf"Wild cluster bootstrap by fiscal year ({int(b.clusters.iloc[0])} clusters), "
              rf"{int(b.B.iloc[0]):,} draws, Webb six-point weights, null imposed, fixed effects "
              r"partialled out; (4)/(5) = column of Table~\ref{tab:main_30d}. Firm: one-way "
              r"firm-clustered $p$ (industry FE, column 5).")
    L += [r"\end{table}"]
    write("inference", L)


def effect_sizes():
    e = pd.read_csv(RES / "effect_sizes_30d.csv")
    L = [r"\begin{table}[htbp]", r"\centering", r"\caption{Economic magnitude of the text coefficients (30-day window)}",
         r"\label{tab:effects}", r"\small", r"\begin{tabular}{llrrrr}", r"\toprule",
         r"FE & Variable & SD pooled & SD within firm & \% per pooled SD & \% per within SD \\", r"\midrule"]
    for _, x in e[e.column == "(5)"].iterrows():
        L.append(f"{'Industry' if x.absorb.startswith('ind') else 'Firm'} + year & {LAB[x['var']]} & "
                 f"{x.sd_pooled:.4f} & {x.sd_within:.4f} & {x.pct_per_sd_pooled:+.2f} & {x.pct_per_sd_within:+.2f}" + r" \\")
    x = e[(e.column == "(4)") & (e["var"] == "risk_1a")]
    for _, y in x.iterrows():
        L.append(f"{'Industry' if y.absorb.startswith('ind') else 'Firm'} + year & RiskDensity (col.\\ 4) & "
                 f"{y.sd_pooled:.4f} & {y.sd_within:.4f} & {y.pct_per_sd_pooled:+.2f} & {y.pct_per_sd_within:+.2f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    L += note(r"Change in annualised 30-day volatility, $100\,(e^{\hat\beta\,\Delta x}-1)$, for an "
              r"increase of one pooled standard deviation or one within-firm standard deviation "
              r"(deviation from the firm mean) of the variable, using the coefficients of "
              r"Table~\ref{tab:main_30d} (industry FE) and Table~\ref{tab:firm_fe} (firm FE).")
    L += [r"\end{table}"]
    write("effect_sizes", L)


def variance_decomp():
    v = pd.read_csv(RES / "variance_decomposition_30d.csv")
    L = [r"\begin{table}[htbp]", r"\centering", r"\caption{Within-firm and between-firm variation}",
         r"\label{tab:variance}", r"\small", r"\begin{tabular}{lrrrrr}", r"\toprule",
         r" & $N$ & SD & SD within & Within share & Within share net of year \\", r"\midrule"]
    for _, x in v.iterrows():
        L.append(f"{LAB[x['var']]} & {n_(x.n)} & {x.sd_total:.4f} & {x.sd_within:.4f} & "
                 f"{x.share_within:.3f} & {x.share_within_net_of_year:.3f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    L += note(r"Within share: variance of the deviation from the firm mean over the total variance; "
              r"the last column first removes fiscal-year means. 30-day estimation sample "
              r"(omission: H3 sample).")
    L += [r"\end{table}"]
    write("variance_decomposition", L)


# ── H3 ──────────────────────────────────────────────────────────────────

def silence_main():
    m = pd.read_csv(RES / "h3_main.csv")
    hs = ["5d", "10d", "30d", "90d", "180d", "365d"]
    L = [r"\begin{table}[htbp]", r"\centering", r"\caption{H3: omission gap and post-filing volatility}",
         r"\label{tab:silence_main}", r"\footnotesize", r"\setlength{\tabcolsep}{3.5pt}",
         r"\begin{tabular}{l" + "c" * len(hs) + "}", r"\toprule",
         " & " + " & ".join(f"{h[:-1]} days" for h in hs) + r" \\", r"\midrule"]
    for meas, title in (("omission_1a", "Omission$^{1A}$"),
                        ("omission_fixedlen_1a", "Length-invariant omission (3{,}000-token blocks)")):
        for an, ft in (("ind+year", "Industry + year FE"), ("firm+year", "Firm + year FE")):
            rr = [m[(m.measure == meas) & (m.h == h) & (m.absorb == an)].iloc[0] for h in hs]
            L += [rf"\multicolumn{{7}}{{l}}{{\emph{{{title}, {ft}}}}} \\",
                  r"$\hat\beta$ & " + " & ".join(cell(r.coef, r.p) for r in rr) + r" \\",
                  " & " + " & ".join(tcell(r.t) for r in rr) + r" \\",
                  r"Holm $p$ & " + " & ".join("--" if pd.isna(r.p_holm) else f"{r.p_holm:.3f}" for r in rr) + r" \\",
                  r"$N$ & " + " & ".join(n_(r.n) for r in rr) + r" \\[3pt]"]
    v = json.loads((RES / "h3_verdict.json").read_text())
    L += [r"\bottomrule", r"\end{tabular}"]
    L += note(r"Each cell is a separate regression of $\ln\sigma^{[h]}_{i,t+1}$ on the omission measure, "
              r"$\ln(\text{Words}^{1A})$, RiskDensity and the common controls. Holm $p$: Holm-adjusted "
              r"over the ten tests of each measure at $h\in\{5,10,30,90,180\}$ under both FE structures "
              r"(365 days is outside the family). Two-way clustered $t$-statistics. "
              rf"Verdict under the decision rule of Section~\ref{{sec:h3}}: {v['verdict']}. " + SIG)
    L += [r"\end{table}"]
    write("silence_main", L)


H3ROB = [("baseline", "Baseline"), ("fixed_length", "Length-invariant omission"),
         ("length_resid", "Residual on $\\ln(\\text{Words}^{1A})$ within cell"),
         ("decile_rank", "Decile rank within cell (cells $\\geq 10$)"),
         ("tail_deciles", "Top-decile dummy (bottom-decile dummy included)"),
         ("peers_ge20", "Cells with $\\geq 20$ peers"),
         ("extra_length", "+ $\\ln(\\text{Words}^{\\text{MD\\&A}})$ and $\\ln(\\text{Words}^{1A})^2$"),
         ("prior_year", "Omission of year $t-1$"),
         ("pooled_vocab", "Pooled vocabulary over all years"),
         ("sic_today", "Peers by today's SIC2"),
         ("plus_lagged_dv", "+ lagged dependent variable")]


def silence_robust():
    m = pd.read_csv(RES / "h3_robust.csv")
    L = [r"\begin{table}[htbp]", r"\centering", r"\caption{H3 robustness (30-day window)}",
         r"\label{tab:silence_robust}", r"\small", r"\begin{tabular}{lcrcr}", r"\toprule",
         r" & \multicolumn{2}{c}{Firm + year FE} & \multicolumn{2}{c}{Industry + year FE} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}", r" & $\hat\beta$ ($t$) & $N$ & $\hat\beta$ ($t$) & $N$ \\",
         r"\midrule"]
    for k, lab in H3ROB:
        cells = []
        for an in ("firm+year", "ind+year"):
            r = m[(m.column == k) & (m.absorb == an)].iloc[0]
            cells += [f"{cell(r.coef, r.p)} {tcell(r.t)}", n_(r.n)]
        L.append(f"{lab} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    L += note(r"H3 specification with one change per row; coefficient on the (modified) omission "
              r"measure. Two-way clustered $t$-statistics. " + SIG)
    L += [r"\end{table}"]
    write("silence_robust", L)


def silence_diag():
    p = pd.read_csv(RES / "h3_placebo.csv").iloc[0]
    d = pd.read_csv(RES / "omission_decomposition.csv").set_index("component")
    g = pd.read_csv(RES / "h3_diagnostics.csv").set_index("stat")
    L = [r"\begin{table}[htbp]", r"\centering", r"\caption{H3 diagnostics: placebo, sources of variation, length}",
         r"\label{tab:silence_diag}", r"\small", r"\begin{tabular}{lr}", r"\toprule",
         r"\multicolumn{2}{l}{\emph{Placebo: random SIC2 cells within year, firm + year FE, 30 days}} \\",
         f"Permutations & {int(p.n_perm):,}" + r" \\",
         f"Observed $t$ & {p.t_obs:.2f}" + r" \\",
         f"Placebo $t$: mean / SD & {p.t_mean:.2f} / {p.t_sd:.2f}" + r" \\",
         f"Placebo $t$: 95th / 97.5th percentile & {p.t_p95:.2f} / {p.t_p975:.2f}" + r" \\",
         f"Empirical $p$ (one-sided / two-sided) & {p.p_emp_one_sided:.3f} / {p.p_emp_two_sided:.3f}" + r" \\",
         r"\midrule",
         rf"\multicolumn{{2}}{{l}}{{\emph{{Decomposition of $\Delta$Omission (n = {int(d.loc['delta','n']):,})}}}} \\"]
    for k, lab in (("own", "Own-text change"), ("peer_text", "Peer-text change"),
                   ("composition", "Peer-composition change"), ("interaction", "Interaction")):
        L.append(f"{lab}: share of variance & {d.loc[k, 'var_share']:.3f}" + r" \\")
    L += [r"\midrule", r"\multicolumn{2}{l}{\emph{Correlation with $\ln(\text{Words}^{1A})$, H3 sample}} \\",
          f"Omission$^{{1A}}$ & {g.loc['corr_omission_1a_log_words_1a','value']:.3f}" + r" \\",
          f"Length-invariant omission & {g.loc['corr_omission_fixedlen_1a_log_words_1a','value']:.3f}" + r" \\",
          f"Pooled-vocabulary omission & {g.loc['corr_omission_pooled_1a_log_words_1a','value']:.3f}" + r" \\",
          f"$N$ & {n_(g.loc['corr_omission_1a_log_words_1a','n'])}" + r" \\",
          r"\bottomrule", r"\end{tabular}"]
    L += note(r"Decomposition in the year-$t$ vocabulary using fiscal-year documents: own = "
              r"$\Omega(o_t,C_{t-1})-\Omega(o_{t-1},C_{t-1})$; peer text = change for the peers present "
              r"in both years; composition = effect of entering/leaving peers; interaction = remainder.")
    L += [r"\end{table}"]
    write("silence_diag", L)


def stationarity():
    s = pd.read_csv(RES / "stationarity.csv")
    lab = dict(LAB, log_vol=r"$\ln\sigma^{[30]}_{i,t+1}$")
    L = [r"\begin{table}[htbp]", r"\centering", r"\caption{Panel unit-root tests}",
         r"\label{tab:stationarity}", r"\footnotesize", r"\begin{tabular}{lrrrrrr}", r"\toprule",
         r" & \multicolumn{3}{c}{Harris--Tzavalis (balanced, 2012--2024)} & \multicolumn{3}{c}{Im--Pesaran--Shin (1 lag)} \\",
         r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r" & $N$ & $\hat\rho$ & $p$ & $N$ & $W_{\bar t}$ & $p$ \\", r"\midrule"]
    for _, x in s.iterrows():
        L.append(f"{lab[x['var']]} & {int(x.ht_N)} & {x.ht_rho:.3f} & {x.ht_p_sim:.3f} & "
                 f"{int(x.ips1_N)} & {x.ips1_W:.2f} & {x.ips1_p:.3f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    L += note(r"Variables cross-sectionally demeaned by fiscal year. Harris--Tzavalis: within estimate "
              r"of $\rho$ on firms observed in every year 2012--2024 ($T=13$); $H_0$: $\rho=1$ for all "
              r"firms; $p$ from 2{,}000 simulated random-walk panels of the same $N$ and $T$. IPS: mean of "
              r"firm ADF $t$-statistics (intercept, one lag; firms with $\geq 8$ consecutive years), "
              r"standardised with simulated null moments; $H_0$: all firms have a unit root, $H_1$: some "
              r"firms are stationary.")
    L += [r"\end{table}"]
    write("stationarity", L)


# ── data appendix ───────────────────────────────────────────────────────

def attrition():
    u = pd.read_csv(DATA / "universe_stages.csv")
    a = pd.read_csv(DATA / "sample_attrition.csv")
    r = pd.read_csv(RES / "regression_sample_attrition_30d.csv")
    L = [r"\begin{table}[htbp]", r"\centering", r"\caption{Sample construction}",
         r"\label{tab:attrition}", r"\small", r"\begin{tabular}{lrr}", r"\toprule",
         r"Stage & Firms & Firm-years \\", r"\midrule",
         r"\multicolumn{3}{l}{\emph{Universe (SEC ticker list, reconstructed with current metadata)}} \\"]
    for _, x in u.iterrows():
        L.append(f"{x.stage.replace('minus ', '$-$ ')} & {int(x.firms):,} & " + r" \\")
    L.append(r"\multicolumn{3}{l}{\emph{Frozen universe (548 firms) to panel}} \\")
    for _, x in a.iloc[2:].iterrows():
        if x.stage in ("raw extraction", "returns & volatility", "ratios", "imputation", "tag locking"):
            continue
        st = (x.stage.replace(">=", "$\\geq$").replace("&", "\\&").replace("<=", "$\\leq$"))
        L.append(f"{st} & {int(x.firms):,} & {n_(x.firm_years)}" + r" \\")
    L.append(r"\multicolumn{3}{l}{\emph{Panel to 30-day estimation sample}} \\")
    for _, x in r.iloc[1:].iterrows():
        st = x.stage.replace(">=", "$\\geq$").replace("_", "\\_")
        L.append(f"{st} & {int(x.firms):,} & {n_(x.firm_years)}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    L += note(r"The 1{,}000 firms are the first 1{,}000 rows of the SEC \texttt{company\_tickers.json} "
              r"list as stored in the repository (2026-10-12); the universe file holds the 548 firms "
              r"that passed these filters in the original run. Each estimation-sample row applies one "
              r"more requirement.")
    L += [r"\end{table}"]
    write("sample_attrition", L)


def tag_tables():
    o = pd.read_csv(DATA / "tag_overlap.csv")
    k = pd.read_csv(DATA / "tag_locking_summary.csv")
    p = pd.read_csv(DATA / "tag_provenance.csv")
    nm = {"net_income": "Net income", "operating_cash_flow": "Operating cash flow",
          "stockholders_equity": "Stockholders' equity"}
    L = [r"\begin{table}[htbp]", r"\centering", r"\caption{XBRL tags: choices, locking losses and overlap}",
         r"\label{tab:tag_provenance}", r"\footnotesize", r"\begin{tabular}{llrr}", r"\toprule",
         r"Metric & Locked tag & Firms & \% \\", r"\midrule"]
    for met, g in p.groupby("metric"):
        vc = g["locked_tag"].value_counts()
        for i, (t, c) in enumerate(vc.items()):
            L.append(f"{nm.get(met, met) if i == 0 else ''} & \\texttt{{{t}}} & {c} & {100 * c / vc.sum():.1f}" + r" \\")
    L += [r"\midrule", r"\multicolumn{4}{l}{\emph{Values set to missing by per-firm tag locking}} \\"]
    for _, x in k.iterrows():
        L.append(f"{nm.get(x.metric, x.metric)} & {int(x.values_nulled):,} of {int(x.values_before):,} & "
                 f"{int(x.firms_with_multiple_tags)} firms with $>1$ tag &" + r" \\")
    L += [r"\midrule", r"\multicolumn{4}{l}{\emph{Paired tags reported together: $|$difference$|$ / total assets (pp)}} \\"]
    for _, x in o.iterrows():
        L.append(f"{nm[x.metric]} & n = {int(x.n_both):,}; median {x.median_pp:.3f}; p95 {x.p95_pp:.2f}; "
                 f"p99 {x.p99_pp:.2f}; max {x.max_pp:.1f} & $>$1\\%: {int(x.n_material)} & "
                 f"{100 * x.share_material:.1f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    L += note(r"Net-income pair: \texttt{ProfitLoss} (consolidated) vs \texttt{NetIncomeLoss} (parent); "
              r"their difference is the non-controlling interest, and the panel uses the consolidated "
              r"figure. Operating-cash-flow pair: total vs continuing operations. ``$>$1\%'': differences "
              r"above 1\% of total assets. Locking: for metrics with non-equivalent tags, each firm keeps "
              r"its most frequent tag.")
    L += [r"\end{table}"]
    write("tag_provenance", L)


def timing_table():
    t = pd.read_csv(RES / "timing_30d.csv").set_index("stat")["value"]
    mon = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    L = [r"\begin{table}[htbp]", r"\centering",
         r"\caption{Fiscal-year ends and filing dates in the estimation sample (30-day window)}",
         r"\label{tab:timing}", r"\small", r"\begin{tabular}{lrr}", r"\toprule",
         r"Month & Fiscal year ends (\%) & 10-K filed (\%) \\", r"\midrule"]
    for m in range(12):
        L.append(f"{mon[m]} & {100 * t[f'share_fye_month_{m + 1}']:.1f} & "
                 f"{100 * t[f'share_filed_in_month_{m + 1}']:.1f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    L += note(r"Shares of the 5{,}782 firm-years. Fiscal-year end month: month of the period end, "
              r"periods ending on 1--7 January counted to December. The median lag between period end "
              rf"and filing is {t['lag_days_median']:.0f} days (10th percentile {t['lag_days_p10']:.0f}, "
              rf"90th percentile {t['lag_days_p90']:.0f}). "
              rf"{int(t['firms_non_dec_fye'])} of the 512 firms have a fiscal year that does not end in "
              rf"December. {100 * t['share_filed_calendar_year_after_fy']:.1f}\% of the 10-Ks are filed in the "
              r"calendar year after the fiscal year, and the filings of one fiscal year span a median of "
              rf"{t['median_calendar_span_of_fy_filings_days']:.0f} calendar days; the filings fall in "
              rf"{int(t['filing_calendar_months_distinct'])} distinct calendar months.")
    L += [r"\end{table}"]
    write("timing", L)


def text_coverage():
    c = pd.read_csv(DATA / "text_coverage_by_year.csv")
    L = [r"\begin{table}[htbp]", r"\centering", r"\caption{Text coverage by fiscal year}",
         r"\label{tab:text_coverage}", r"\small", r"\begin{tabular}{lrrr}", r"\toprule",
         r"Fiscal year & Filings & Item~1A (\%) & Item~7 (\%) \\", r"\midrule"]
    for _, x in c.iterrows():
        fy = "All" if str(x.fiscal_year) == "all" else str(x.fiscal_year)
        if fy == "All":
            L.append(r"\midrule")
        L.append(f"{fy} & {int(x.filings):,} & {100 * x.item_1a_rate:.1f} & {100 * x.item_7_rate:.1f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    L += note(r"Share of panel firm-years whose stored section has at least 100 tokens.")
    L += [r"\end{table}"]
    write("text_coverage", L)


def h3_cells():
    c = pd.read_csv(RES / "h3_peers_ge20_composition.csv")
    L = [r"\begin{table}[htbp]", r"\centering", r"\caption{Industries in the $\geq 20$-peer H3 sample}",
         r"\label{tab:h3_cells}", r"\small", r"\begin{tabular}{lrrrr}", r"\toprule",
         r"SIC2 & Firm-years & Firms & Share & Share of full H3 sample \\", r"\midrule"]
    for _, x in c.iterrows():
        L.append(f"{int(x['ind'])} & {int(x.firm_years):,} & {int(x.firms)} & {x.share:.3f} & {x.share_full_h3:.3f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    L += note(r"SIC2 = two-digit SIC code at filing (e.g.\ 28 chemicals and pharmaceuticals, 35 industrial "
              r"machinery and computers, 36 electronic equipment, 38 instruments, 49 utilities, "
              r"73 business services).")
    L += [r"\end{table}"]
    write("h3_cells", L)


def main():
    for f in (summary, correlation, lambda: main_table("30d"), lambda: main_table("365d"),
              horizons, firm_fe, lambda: robustness("30d"), lambda: robustness("365d"),
              inference, effect_sizes, variance_decomp, silence_main, silence_robust,
              silence_diag, stationarity, attrition, tag_tables, text_coverage, h3_cells,
              timing_table):
        f()


if __name__ == "__main__":
    main()

"""Regression analysis for the Volume--Specificity--Silence volatility study.

Estimates two nested hypotheses on the disclosed-content side
(H1 + H2 with the LM-list decomposition as the second stage of H2),
jointly with financial controls and SIC2 + year fixed effects, with
firm-clustered standard errors. The strategic-silence hypothesis (H3)
is estimated separately by ``scripts/silence_table.py``.

  H1 — Volume       : ``log_words_1a``  (length of Item~1A)
  H2 — Specificity  : ``risk_1a``       (LM uncertainty + litigious density)
                       Second-stage LM-list decomposition splits ``risk_1a``
                       into ``unc_1a`` (LM uncertainty density) and
                       ``lit_1a`` (LM litigious density); the negative
                       effect on the composite is shown to be carried by
                       the litigious component.

Produces the LaTeX tables consumed by ``tex/chapters/05_results.tex`` and
``tex/chapters/06_robustness.tex``.

The dependent variable is ``log(vol_h)``: the natural log of the
annualised standard deviation of daily log returns over a post-filing
window of ``h`` calendar days. Coefficients on continuous regressors
are therefore semi-elasticities.

Multi-horizon support reuses the same specification at
``h in {5, 10, 30, 90, 180, 365}`` so that the time-stability of the
disclosure-volatility relation can be documented.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.iv.absorbing import AbsorbingLS

from config import DATA_DIR

BASE_DIR = Path(__file__).resolve().parent.parent
TEX_TABLE_DIR = BASE_DIR / "tex" / "tables"
TEX_TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ── Variable labels for LaTeX output ──────────────────────────────────────

LABELS = {
    "log_vol":          r"$\ln(\sigma_{i,t+1})$",
    "vol_next_year":    r"$\sigma_{i,t+1}$",
    "lagged_log_vol":   r"$\ln(\sigma_{i,t})$",
    "lagged_vol":       r"$\sigma_{i,t}$",
    "log_total_assets": r"$\ln(\text{Total Assets})$",
    "leverage":         "Leverage",
    "roa":              "ROA",
    "asset_growth":     "Asset Growth",
    "log_words_1a":     r"$\ln(\text{Words}^{\text{1A}})$",
    "risk_1a":          r"RiskDensity",
    "unc_1a":           r"UncDensity",
    "lit_1a":           r"LitDensity",
    "sentiment_mda":    r"Sentiment$^{\text{MD\&A}}$",
}

# Block of financial controls reused in every specification.
FIN_VARS = ["log_total_assets", "leverage", "roa", "asset_growth"]

# Three text constructs of the headline test.
LEN_VAR  = "log_words_1a"   # H1: verbosity of Item 1A
DENS_VAR = "risk_1a"        # H2: composite LM risk-word density of Item 1A
UNC_VAR  = "unc_1a"         # H2 second stage: LM-Uncertainty density
LIT_VAR  = "lit_1a"         # H2 second stage: LM-Litigious   density

# Variables for descriptive panel.
PANEL_VARS = [
    "log_vol", "lagged_log_vol", "log_total_assets",
    "leverage", "roa", "asset_growth",
    "log_words_1a", "risk_1a", "unc_1a", "lit_1a",
]

# Multi-horizon configuration: (label, vol col, lagged vol col).
HORIZONS: list[tuple[str, str, str]] = [
    ("5d",   "vol_5d",        "lagged_vol_5d"),
    ("10d",  "vol_10d",       "lagged_vol_10d"),
    ("30d",  "vol_30d",       "lagged_vol_30d"),
    ("90d",  "vol_90d",       "lagged_vol_90d"),
    ("180d", "vol_180d",      "lagged_vol_180d"),
    ("365d", "vol_next_year", "lagged_vol"),
]


# ── Data loading ──────────────────────────────────────────────────────────

def _add_log_vol(df: pd.DataFrame, vol_col: str, lag_col: str) -> pd.DataFrame:
    """Return a copy with log(vol) and lagged log(vol) added."""
    out = df.copy()
    out["log_vol"]        = np.log(out[vol_col].clip(lower=1e-6))
    out["lagged_log_vol"] = np.log(out[lag_col].clip(lower=1e-6))
    return out


def load_text_panel(vol_col: str = "vol_next_year",
                    lag_col: str = "lagged_vol") -> pd.DataFrame:
    """Load the text-extended panel and prune to rows usable for one horizon.

    Substitutes ``lag_col`` for the AR(1) control so each horizon uses its
    own lagged volatility. Drops rows missing any regression variable.
    Adds the SIC-2 industry code, the log(vol) dependent variable and its
    lagged counterpart.
    """
    df = pd.read_csv(DATA_DIR / "annual_panel_text.csv")
    if vol_col not in df.columns or lag_col not in df.columns:
        raise FileNotFoundError(
            f"{vol_col} or {lag_col} not in panel — "
            "run build_volatility_horizons.py first."
        )
    df["sic2"] = (df["sic"] // 100).astype(int)
    if "log_words_1a" not in df.columns:
        df["log_words_1a"] = np.log(df["words_1a"].clip(lower=1))
    needed = [vol_col, lag_col, *FIN_VARS, LEN_VAR, DENS_VAR,
              UNC_VAR, LIT_VAR,
              "sic2", "fiscal_year", "ticker"]
    df = df[needed].dropna().copy()
    df = _add_log_vol(df, vol_col, lag_col)
    return df


# ── Estimation helpers ────────────────────────────────────────────────────

def _fit(df: pd.DataFrame, xvars: list[str]):
    """Fit one absorbing-LS model with industry+year FE and two-way
    (firm + year) clustered standard errors."""
    y = df["log_vol"]
    X = df[xvars].copy()
    X.insert(0, "const", 1.0)
    absorb = df[["sic2", "fiscal_year"]].astype("category")
    clusters = df[["ticker", "fiscal_year"]].astype("category").apply(
        lambda c: c.cat.codes
    )
    return AbsorbingLS(y, X, absorb=absorb).fit(
        cov_type="clustered", clusters=clusters,
    )


def run_main(df: pd.DataFrame | None = None,
             vol_col: str = "vol_next_year",
             lag_col: str = "lagged_vol"):
    """Estimate the five nested specifications for the headline table.

    Columns are:
        (1) baseline financial controls
        (2) +H1                       : log_words_1a
        (3) +H2                       : composite risk_1a
        (4) +H1 + H2                  : both jointly
        (5) +H1 + H3 (decomposition)  : log_words_1a + unc_1a + lit_1a
    """
    if df is None:
        df = load_text_panel(vol_col=vol_col, lag_col=lag_col)
    base = ["lagged_log_vol", *FIN_VARS]
    specs = [
        list(base),
        list(base) + [LEN_VAR],
        list(base) + [DENS_VAR],
        list(base) + [LEN_VAR, DENS_VAR],
        list(base) + [LEN_VAR, UNC_VAR, LIT_VAR],
    ]
    results = [_fit(df, x) for x in specs]
    return results, df, specs


def run_horizons():
    """Estimate the H1 + H3-decomposition model on every available horizon."""
    out = []
    for label, vol_col, lag_col in HORIZONS:
        try:
            df = load_text_panel(vol_col=vol_col, lag_col=lag_col)
        except FileNotFoundError as e:
            print(f"  [skip] horizon {label}: {e}")
            continue
        if df.empty:
            print(f"  [skip] horizon {label}: empty after dropna")
            continue
        xvars = ["lagged_log_vol", *FIN_VARS, LEN_VAR, UNC_VAR, LIT_VAR]
        res = _fit(df, xvars)
        out.append((label, res, df))
    return out


# ── (Removed) Decay-of-information half-life fit ──────────────────────────
# The earlier H3 framing as an information half-life was superseded by the
# H3 boilerplate-decomposition (see ``run_main`` columns 3-5 and the
# horizons table). The half-life code and its LaTeX writer are no longer
# needed.


# ── LaTeX table formatting ────────────────────────────────────────────────

def _stars(pval: float) -> str:
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.10:
        return "*"
    return ""


def main_table_to_latex(results, df, specs,
                        path: Path | None = None,
                        caption: str | None = None,
                        label: str = "tab:main_30d",
                        horizon_label: str = "30 days") -> str:
    """Headline four-column nested table for the main horizon."""
    n_cols = len(results)
    # Union of all variables across specs, preserving first-seen order so
    # that a variable that appears only in some columns still gets its row.
    all_vars: list[str] = []
    for x in specs:
        for v in x:
            if v not in all_vars:
                all_vars.append(v)
    n_firms = df["ticker"].nunique()

    if caption is None:
        caption = (rf"Item~1A Length, Risk-Word Density, and the "
                   rf"Boilerplate Decomposition ({horizon_label} window)")

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{l" + "c" * n_cols + "}")
    lines.append(r"\toprule")
    header = " & ".join([f"({i+1})" for i in range(n_cols)])
    lines.append(f" & {header} \\\\")
    lines.append(r"\midrule")

    for var in all_vars:
        label_v = LABELS.get(var, var)
        coefs, tstats = [], []
        for res, xvars in zip(results, specs):
            if var in xvars:
                c = res.params[var]
                t = res.tstats[var]
                p = res.pvalues[var]
                coefs.append(f"${c:+.4f}${_stars(p)}")
                tstats.append(f"$({t:+.2f})$")
            else:
                coefs.append("")
                tstats.append("")
        lines.append(f"{label_v} & " + " & ".join(coefs) + r" \\")
        lines.append(" & " + " & ".join(tstats) + r" \\[4pt]")

    lines.append(r"\midrule")
    lines.append("Industry FE & " + " & ".join(["Yes"] * n_cols) + r" \\")
    lines.append("Year FE & "     + " & ".join(["Yes"] * n_cols) + r" \\")
    obs   = [f"{int(res.nobs):,}"        for res in results]
    firms = [str(n_firms)] * n_cols
    r2    = [f"{res.rsquared_adj:.3f}"   for res in results]
    lines.append("Observations & " + " & ".join(obs)   + r" \\")
    lines.append("Firms & "        + " & ".join(firms) + r" \\")
    lines.append(r"Adj.\ $R^2$ & " + " & ".join(r2)    + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\par\smallskip")
    lines.append(r"\noindent\footnotesize ")
    lines.append(
        r"\textit{Note:} "
        r"The dependent variable is $\ln(\sigma_{i,t+1})$, the natural "
        rf"log of annualised post-filing volatility over a {horizon_label}~"
        r"window starting two trading days after the 10-K filing date. "
        r"Column~(1) reports the financial baseline; column~(2) adds the log "
        r"word count of Item~1A (H1); column~(3) adds the composite LM "
        r"risk-word density (H2); column~(4) adds H1 and H2 jointly; "
        r"column~(5) replaces the H2 composite by its LM-list decomposition into "
        r"the LM-Uncertainty and LM-Litigious densities of Item~1A. All "
        r"specifications include two-digit SIC industry and fiscal-year "
        r"fixed effects. $t$-statistics in parentheses use two-way "
        r"(firm and fiscal-year) clustered "
        r"standard errors. ***, **, * denote significance at the 1\%, 5\%, "
        r"and 10\% levels."
    )
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    if path is not None:
        path.write_text(tex, encoding="utf-8")
        print(f"  Table saved → {path.name}")
    return tex


def horizons_to_latex(runs, path: Path | None = None) -> str:
    """One column per horizon for the joint H1+H2 specification."""
    if not runs:
        return ""

    show = [
        ("lagged_log_vol", LABELS["lagged_log_vol"]),
        (LEN_VAR,          LABELS[LEN_VAR]),
        (UNC_VAR,          LABELS[UNC_VAR]),
        (LIT_VAR,          LABELS[LIT_VAR]),
    ]
    n_cols = len(runs)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Post-Filing Volatility at Alternative Horizons}")
    lines.append(r"\label{tab:horizons}")
    lines.append(r"\begin{tabular}{l" + "c" * n_cols + "}")
    lines.append(r"\toprule")
    header = " & ".join(f"({i+1}) {label}" for i, (label, _, _) in enumerate(runs))
    lines.append(f" & {header} \\\\")
    lines.append(r"\midrule")

    for var, label_v in show:
        coefs, tstats = [], []
        for _, res, _ in runs:
            if var not in res.params.index:
                coefs.append("")
                tstats.append("")
                continue
            c = res.params[var]; t = res.tstats[var]; p = res.pvalues[var]
            coefs.append(f"${c:+.4f}${_stars(p)}")
            tstats.append(f"$({t:+.2f})$")
        lines.append(f"{label_v} & " + " & ".join(coefs) + r" \\")
        lines.append(" & " + " & ".join(tstats) + r" \\[4pt]")

    lines.append(r"\midrule")
    lines.append("Financial controls & " + " & ".join(["Yes"] * n_cols) + r" \\")
    lines.append("Industry FE & "        + " & ".join(["Yes"] * n_cols) + r" \\")
    lines.append("Year FE & "            + " & ".join(["Yes"] * n_cols) + r" \\")
    obs   = " & ".join(f"{int(res.nobs):,}"        for _, res, _   in runs)
    firms = " & ".join(f"{df['ticker'].nunique()}" for _, _,   df  in runs)
    r2    = " & ".join(f"{res.rsquared_adj:.3f}"   for _, res, _   in runs)
    lines.append("Observations & " + obs   + r" \\")
    lines.append("Firms & "        + firms + r" \\")
    lines.append(r"Adj.\ $R^2$ & " + r2    + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\par\smallskip")
    lines.append(r"\noindent\footnotesize ")
    lines.append(
        r"\textit{Note:} "
        r"Each column re-estimates the H1 + LM-list decomposition specification "
        r"replacing the dependent variable with $\ln(\sigma_{i,t+1}^{[h]})$ "
        r"measured over a post-filing window of length $h$. The lagged "
        r"dependent variable is the same firm's log-volatility computed over "
        r"the prior filing year's window of identical length. UncDensity "
        r"and LitDensity are the LM-Uncertainty and LM-Litigious word "
        r"densities on Item~1A; together they sum to RiskDensity (H2). "
        r"Financial controls (size, leverage, ROA, asset growth) are "
        r"included but suppressed. Industry and year fixed effects are "
        r"absorbed; $t$-statistics in parentheses use two-way "
        r"(firm and fiscal-year) clustered "
        r"standard errors. ***, **, * denote significance at the 1\%, 5\%, "
        r"and 10\% levels."
    )
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    if path is not None:
        path.write_text(tex, encoding="utf-8")
        print(f"  Table saved → {path.name}")
    return tex


def half_life_to_latex(*args, **kwargs) -> str:
    """Removed. Kept as a no-op shim in case external callers import it."""
    return ""


# ── Descriptive statistics ────────────────────────────────────────────────

def descriptive_stats_to_latex(df: pd.DataFrame,
                               path: Path | None = None) -> str:
    stats = df[PANEL_VARS].describe(percentiles=[0.25, 0.50, 0.75]).T
    stats = stats[["mean", "std", "25%", "50%", "75%"]]
    stats.columns = ["Mean", "SD", "P25", "Median", "P75"]

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Summary Statistics}")
    lines.append(r"\label{tab:summary}")
    lines.append(r"\begin{tabular}{lccccc}")
    lines.append(r"\toprule")
    lines.append(r" & Mean & SD & P25 & Median & P75 \\")
    lines.append(r"\midrule")
    for var in PANEL_VARS:
        label = LABELS.get(var, var)
        row = stats.loc[var]
        lines.append(
            f"{label} & {row['Mean']:.3f} & {row['SD']:.3f} & "
            f"{row['P25']:.3f} & {row['Median']:.3f} & {row['P75']:.3f} \\\\"
        )
    lines.append(r"\midrule")
    lines.append(rf"Observations & \multicolumn{{5}}{{c}}{{{len(df):,}}} \\")
    lines.append(rf"Firms & \multicolumn{{5}}{{c}}{{{df['ticker'].nunique()}}} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\par\smallskip")
    lines.append(r"\noindent\footnotesize ")
    lines.append(
        r"\textit{Note:} "
        r"Summary statistics for the regression sample at the 30-day "
        r"horizon. $\sigma_{i,t+1}$ is the annualised standard deviation "
        r"of daily log returns over the 30-day window starting two trading "
        r"days after the filing date; $\ln(\sigma_{i,t+1})$ is its natural "
        r"logarithm. Words$^{\text{1A}}$ is the cleaned word count of Item~1A; "
        r"RiskDensity is the share of LM uncertainty- and litigious-list "
        r"words in Item~1A; UncDensity and LitDensity are the two "
        r"sub-components separately. Financial ratios are winsorised at the "
        r"1st and 99th percentiles."
    )
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    if path is not None:
        path.write_text(tex, encoding="utf-8")
        print(f"  Table saved → {path.name}")
    return tex


def correlation_to_latex(df: pd.DataFrame,
                         path: Path | None = None) -> str:
    from scipy.stats import pearsonr
    n = len(PANEL_VARS)
    corr = pd.DataFrame(np.nan, index=PANEL_VARS, columns=PANEL_VARS)
    pvals = pd.DataFrame(np.nan, index=PANEL_VARS, columns=PANEL_VARS)
    for i in range(n):
        for j in range(n):
            r, p = pearsonr(df[PANEL_VARS[i]], df[PANEL_VARS[j]])
            corr.iloc[i, j]  = r
            pvals.iloc[i, j] = p

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Pearson Correlation Matrix}")
    lines.append(r"\label{tab:correlation}")
    col_spec = "l" + "c" * n
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")
    header = " & ".join([f"({i+1})" for i in range(n)])
    lines.append(f" & {header} \\\\")
    lines.append(r"\midrule")
    for i in range(n):
        label = LABELS.get(PANEL_VARS[i], PANEL_VARS[i])
        cells = []
        for j in range(n):
            if j > i:
                cells.append("")
            elif i == j:
                cells.append("1")
            else:
                r = corr.iloc[i, j]
                p = pvals.iloc[i, j]
                cells.append(f"{r:.2f}{_stars(p)}")
        lines.append(f"({i+1}) {label} & " + " & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\par\smallskip")
    lines.append(r"\noindent\footnotesize ")
    lines.append(
        r"\textit{Note:} "
        r"Pairwise Pearson correlation coefficients on the 30-day-horizon "
        r"regression sample. ***, **, * denote significance at the 1\%, 5\%, "
        r"and 10\% levels."
    )
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    if path is not None:
        path.write_text(tex, encoding="utf-8")
        print(f"  Table saved → {path.name}")
    return tex


# ── Robustness: pre/post-COVID split ─────────────────────────────────────

def run_subperiods(vol_col: str = "vol_30d",
                   lag_col: str = "lagged_vol_30d") -> dict:
    """Estimate the H1 + H3-decomposition model on pre/post-COVID samples."""
    df = load_text_panel(vol_col=vol_col, lag_col=lag_col)
    out = {}
    for tag, mask in [("pre",  df["fiscal_year"] <= 2019),
                      ("post", df["fiscal_year"] >= 2020)]:
        sub = df[mask].copy()
        if sub["fiscal_year"].nunique() < 2:
            continue
        xvars = ["lagged_log_vol", *FIN_VARS, LEN_VAR, UNC_VAR, LIT_VAR]
        out[tag] = (_fit(sub, xvars), sub)
    return out


def subperiod_to_latex(results: dict, path: Path | None = None) -> str:
    if not results:
        return ""
    cols = list(results.keys())
    show = [
        ("lagged_log_vol", LABELS["lagged_log_vol"]),
        (LEN_VAR,          LABELS[LEN_VAR]),
        (UNC_VAR,          LABELS[UNC_VAR]),
        (LIT_VAR,          LABELS[LIT_VAR]),
    ]
    n_cols = len(cols)
    pretty = {"pre": "Pre-COVID (2010--2019)",
              "post": "COVID era (2020--2024)"}

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Sub-Period Robustness (30-day horizon)}")
    lines.append(r"\label{tab:subperiod}")
    lines.append(r"\begin{tabular}{l" + "c" * n_cols + "}")
    lines.append(r"\toprule")
    header = " & ".join(f"({i+1}) {pretty[c]}" for i, c in enumerate(cols))
    lines.append(f" & {header} \\\\")
    lines.append(r"\midrule")

    for var, label_v in show:
        coefs, tstats = [], []
        for c in cols:
            res, _ = results[c]
            v = res.params[var]; t = res.tstats[var]; p = res.pvalues[var]
            coefs.append(f"${v:+.4f}${_stars(p)}")
            tstats.append(f"$({t:+.2f})$")
        lines.append(f"{label_v} & " + " & ".join(coefs) + r" \\")
        lines.append(" & " + " & ".join(tstats) + r" \\[4pt]")

    lines.append(r"\midrule")
    lines.append("Financial controls & " + " & ".join(["Yes"] * n_cols) + r" \\")
    lines.append("Industry FE & "        + " & ".join(["Yes"] * n_cols) + r" \\")
    lines.append("Year FE & "            + " & ".join(["Yes"] * n_cols) + r" \\")
    obs   = " & ".join(f"{int(results[c][0].nobs):,}" for c in cols)
    firms = " & ".join(f"{results[c][1]['ticker'].nunique()}" for c in cols)
    r2    = " & ".join(f"{results[c][0].rsquared_adj:.3f}"   for c in cols)
    lines.append("Observations & " + obs + r" \\")
    lines.append("Firms & "        + firms + r" \\")
    lines.append(r"Adj.\ $R^2$ & " + r2    + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\par\smallskip")
    lines.append(r"\noindent\footnotesize ")
    lines.append(
        r"\textit{Note:} "
        r"H1 + LM-list decomposition specification re-estimated on two "
        r"non-overlapping sub-samples. The dependent variable is "
        r"$\ln(\sigma_{i,t+1}^{[30]})$. UncDensity and LitDensity are the "
        r"LM-Uncertainty and LM-Litigious word densities on Item~1A. All "
        r"financial controls and SIC2$+$year fixed effects are included. "
        r"Standard errors are two-way clustered by firm and fiscal year. ***, **, * denote significance "
        r"at the 1\%, 5\%, and 10\% levels."
    )
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    if path is not None:
        path.write_text(tex, encoding="utf-8")
        print(f"  Table saved → {path.name}")
    return tex


# ── Robustness: firm fixed effects ───────────────────────────────────────

def run_firm_fe(vol_col: str = "vol_30d",
                lag_col: str = "lagged_vol_30d"):
    """Re-estimate the H1 + H3-decomposition model with firm + year FE."""
    df = load_text_panel(vol_col=vol_col, lag_col=lag_col)
    y = df["log_vol"]
    X = df[["lagged_log_vol", *FIN_VARS, LEN_VAR, UNC_VAR, LIT_VAR]].copy()
    X.insert(0, "const", 1.0)
    absorb = df[["ticker", "fiscal_year"]].astype("category")
    res = AbsorbingLS(y, X, absorb=absorb).fit(
        cov_type="clustered", clusters=df["ticker"],
    )
    return res, df


def firm_fe_to_latex(res, df, path: Path | None = None) -> str:
    show = [
        ("lagged_log_vol", LABELS["lagged_log_vol"]),
        (LEN_VAR,          LABELS[LEN_VAR]),
        (UNC_VAR,          LABELS[UNC_VAR]),
        (LIT_VAR,          LABELS[LIT_VAR]),
    ]
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Within-Firm Identification (Firm $+$ Year FE, 30-day horizon)}")
    lines.append(r"\label{tab:firm_fe}")
    lines.append(r"\begin{tabular}{lc}")
    lines.append(r"\toprule")
    lines.append(r" & Estimate \\")
    lines.append(r"\midrule")
    for var, label_v in show:
        c = res.params[var]; t = res.tstats[var]; p = res.pvalues[var]
        lines.append(f"{label_v} & ${c:+.4f}${_stars(p)} \\\\")
        lines.append(f" & $({t:+.2f})$ \\\\[4pt]")
    lines.append(r"\midrule")
    lines.append(r"Financial controls & Yes \\")
    lines.append(r"Firm FE & Yes \\")
    lines.append(r"Year FE & Yes \\")
    lines.append(rf"Observations & {int(res.nobs):,} \\")
    lines.append(rf"Firms & {df['ticker'].nunique()} \\")
    lines.append(rf"Adj.\ $R^2$ & {res.rsquared_adj:.3f} \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\par\smallskip")
    lines.append(r"\noindent\footnotesize ")
    lines.append(
        r"\textit{Note:} "
        r"H1 + LM-list decomposition specification with firm and fiscal-year "
        r"fixed effects absorbed. Identification of $\ln(\text{Words}^{1A})$, "
        r"UncDensity and LitDensity comes only from within-firm variation "
        r"across years. Standard errors clustered by firm. ***, **, * denote "
        r"significance at the 1\%, 5\%, and 10\% levels."
    )
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    if path is not None:
        path.write_text(tex, encoding="utf-8")
        print(f"  Table saved → {path.name}")
    return tex


# ── Driver ────────────────────────────────────────────────────────────────

def _print_result(res, vars_to_show: list[str]) -> None:
    print(f"  N = {int(res.nobs):,}  Adj R² = {res.rsquared_adj:.4f}")
    for var in vars_to_show:
        if var not in res.params.index:
            continue
        c, t, p = res.params[var], res.tstats[var], res.pvalues[var]
        print(f"    {var:18s}  β = {c:+10.4f}  t = {t:+6.2f}  p = {p:.4f} {_stars(p)}")


def main() -> None:
    print("=== Loading panel at primary horizon (30-day) ===")
    df30 = load_text_panel(vol_col="vol_30d", lag_col="lagged_vol_30d")
    print(f"  {df30['ticker'].nunique()} firms, {len(df30):,} firm-years")

    print("\n=== Summary statistics + correlation matrix (30d sample) ===")
    descriptive_stats_to_latex(df30, TEX_TABLE_DIR / "summary_statistics.tex")
    correlation_to_latex(df30,       TEX_TABLE_DIR / "correlation_matrix.tex")

    print("\n=== Headline table (30-day horizon) ===")
    res30, _, specs30 = run_main(df30)
    for i, r in enumerate(res30, 1):
        print(f"\n  Spec ({i}): {', '.join(specs30[i-1])}")
        _print_result(r, specs30[i-1])
    main_table_to_latex(res30, df30, specs30,
                        path=TEX_TABLE_DIR / "main_30d.tex",
                        horizon_label="30 days",
                        label="tab:main_30d")

    # ── H3 Wald test: β^unc = β^lit on the headline (column 5) ──
    h3_res = res30[-1]
    try:
        import numpy as np
        b_unc = h3_res.params[UNC_VAR]
        b_lit = h3_res.params[LIT_VAR]
        cov = h3_res.cov
        var_diff = (cov.loc[UNC_VAR, UNC_VAR]
                    + cov.loc[LIT_VAR, LIT_VAR]
                    - 2 * cov.loc[UNC_VAR, LIT_VAR])
        se_diff = float(np.sqrt(var_diff))
        diff = float(b_unc - b_lit)
        z = diff / se_diff
        # two-sided p-value from standard normal
        from scipy.stats import norm
        p = 2 * (1 - norm.cdf(abs(z)))
        # 95% CI on β^unc
        se_unc = float(np.sqrt(cov.loc[UNC_VAR, UNC_VAR]))
        ci_lo = b_unc - 1.96 * se_unc
        ci_hi = b_unc + 1.96 * se_unc
        print(f"\n  H3 Wald test  H0: β^unc = β^lit")
        print(f"    diff = {diff:+.3f}  se = {se_diff:.3f}  "
              f"z = {z:+.2f}  p = {p:.4f}")
        print(f"  95% CI on β^unc: [{ci_lo:+.2f}, {ci_hi:+.2f}]")
    except Exception as exc:  # pragma: no cover
        print(f"  Wald test failed: {exc}")

    print("\n=== Headline table (365-day horizon) ===")
    df365 = load_text_panel(vol_col="vol_next_year", lag_col="lagged_vol")
    res365, _, specs365 = run_main(df365)
    for i, r in enumerate(res365, 1):
        print(f"\n  Spec ({i}): {', '.join(specs365[i-1])}")
        _print_result(r, specs365[i-1])
    main_table_to_latex(res365, df365, specs365,
                        path=TEX_TABLE_DIR / "main_365d.tex",
                        horizon_label="365 days",
                        label="tab:main_365d")

    print("\n=== Multi-horizon table ===")
    runs = run_horizons()
    for label, res, _ in runs:
        bL = res.params[LEN_VAR]; tL = res.tstats[LEN_VAR]
        bU = res.params[UNC_VAR]; tU = res.tstats[UNC_VAR]
        bI = res.params[LIT_VAR]; tI = res.tstats[LIT_VAR]
        print(f"  {label:>5s}: N={int(res.nobs):>5,}  "
              f"Adj R²={res.rsquared_adj:.3f}  "
              f"len β={bL:+.4f} (t={tL:+.2f})  "
              f"unc β={bU:+.3f} (t={tU:+.2f})  "
              f"lit β={bI:+.3f} (t={tI:+.2f})")
    horizons_to_latex(runs, path=TEX_TABLE_DIR / "horizons.tex")

    print("\n=== Sub-period robustness ===")
    subres = run_subperiods()
    for tag, (res, sub) in subres.items():
        print(f"  {tag:>4s}: N={int(res.nobs):>5,}  "
              f"len β={res.params[LEN_VAR]:+.4f} (t={res.tstats[LEN_VAR]:+.2f})  "
              f"unc β={res.params[UNC_VAR]:+.3f} (t={res.tstats[UNC_VAR]:+.2f})  "
              f"lit β={res.params[LIT_VAR]:+.3f} (t={res.tstats[LIT_VAR]:+.2f})")
    subperiod_to_latex(subres, path=TEX_TABLE_DIR / "subperiods.tex")

    print("\n=== Firm-FE robustness ===")
    res_ffe, df_ffe = run_firm_fe()
    print(f"  N = {int(res_ffe.nobs):,}  Adj R² = {res_ffe.rsquared_adj:.4f}")
    for v in [LEN_VAR, UNC_VAR, LIT_VAR]:
        print(f"    {v:18s}  β = {res_ffe.params[v]:+.4f}  "
              f"t = {res_ffe.tstats[v]:+.2f}")
    firm_fe_to_latex(res_ffe, df_ffe, path=TEX_TABLE_DIR / "firm_fe.tex")

    # Drop any stale half-life table left over from previous runs.
    stale = TEX_TABLE_DIR / "half_life.tex"
    if stale.exists():
        stale.unlink()
        print(f"  Removed stale {stale.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()

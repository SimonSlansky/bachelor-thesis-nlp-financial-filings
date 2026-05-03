"""Regression analysis for volatility determinants.

Produces LaTeX tables for the Results chapter.  Each public function
returns fitted model objects so they can be inspected interactively.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from linearmodels.iv.absorbing import AbsorbingLS

from config import DATA_DIR

BASE_DIR = Path(__file__).resolve().parent.parent
TEX_TABLE_DIR = BASE_DIR / "tex" / "tables"
TEX_TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ── Variable labels for LaTeX output ──────────────────────────────────────

LABELS = {
    "vol_next_year": r"Volatility$_{t+1}$",
    "lagged_vol": "Lagged Volatility",
    "log_total_assets": r"$\ln(\text{Total Assets})$",
    "leverage": "Leverage",
    "roa": "ROA",
    "asset_growth": "Asset Growth",
    "delta_sentiment": r"$\Delta$Sentiment",
    "delta_risk": r"$\Delta$Risk",
    "textsim": "TextSim",
    "divergence": "Divergence",
}

# Variable order for summary / correlation tables
PANEL_VARS = [
    "vol_next_year", "lagged_vol", "log_total_assets",
    "leverage", "roa", "asset_growth",
]

# Financial-control block reused across baseline / text / divergence models.
FIN_VARS = ["lagged_vol", "log_total_assets", "leverage", "roa", "asset_growth"]
TEXT_VARS = ["delta_sentiment", "delta_risk", "textsim"]


# ── Data loading ──────────────────────────────────────────────────────────

def load_regression_panel() -> pd.DataFrame:
    """Load panel, add SIC-2, drop rows with missing regression variables."""
    df = pd.read_csv(DATA_DIR / "annual_panel.csv")
    df["sic2"] = (df["sic"] // 100).astype(int)

    needed = [
        "vol_next_year", "lagged_vol", "log_total_assets",
        "leverage", "roa", "asset_growth",
        "sic2", "fiscal_year", "ticker",
    ]
    df = df[needed].dropna().copy()
    return df


def load_text_panel() -> pd.DataFrame:
    """Load the text-extended panel and drop rows missing any text feature."""
    df = pd.read_csv(DATA_DIR / "annual_panel_text.csv")
    df["sic2"] = (df["sic"] // 100).astype(int)
    needed = [
        "vol_next_year", "lagged_vol", "log_total_assets",
        "leverage", "roa", "asset_growth",
        "delta_sentiment", "delta_risk", "textsim", "divergence",
        "sic2", "fiscal_year", "ticker",
    ]
    df = df[needed].dropna().copy()
    return df


# ── Regression runner ─────────────────────────────────────────────────────

def run_baseline(df: pd.DataFrame | None = None):
    """Estimate three nested baseline regressions.

    (1) Lagged vol only  + Industry FE + Year FE
    (2) + Size, Leverage
    (3) + ROA, Asset Growth  (full baseline)

    Returns (results, df, specs).
    """
    if df is None:
        df = load_regression_panel()

    y = df["vol_next_year"]
    absorb = df[["sic2", "fiscal_year"]].astype("category")
    clusters = df["ticker"]

    specs = [
        ["lagged_vol"],
        ["lagged_vol", "log_total_assets", "leverage"],
        ["lagged_vol", "log_total_assets", "leverage", "roa", "asset_growth"],
    ]

    results = []
    for xvars in specs:
        X = df[xvars].copy()
        model = AbsorbingLS(y, X, absorb=absorb)
        res = model.fit(cov_type="clustered", clusters=clusters)
        results.append(res)

    return results, df, specs


def _fit(df: pd.DataFrame, xvars: list[str], dep: str = "vol_next_year"):
    """Fit one absorbing-LS model with industry+year FE and firm clusters."""
    y = df[dep]
    X = df[xvars].copy()
    absorb = df[["sic2", "fiscal_year"]].astype("category")
    clusters = df["ticker"]
    return AbsorbingLS(y, X, absorb=absorb).fit(
        cov_type="clustered", clusters=clusters,
    )


def run_text(df: pd.DataFrame | None = None):
    """Estimate three nested specifications culminating in the text model.

    (1) Baseline      : financial controls only.
    (2) +Sentiment+Risk
    (3) Full text model: + TextSim.
    """
    if df is None:
        df = load_text_panel()
    specs = [
        list(FIN_VARS),
        list(FIN_VARS) + ["delta_sentiment", "delta_risk"],
        list(FIN_VARS) + list(TEXT_VARS),
    ]
    results = [_fit(df, x) for x in specs]
    return results, df, specs


def run_divergence(df: pd.DataFrame | None = None):
    """Estimate three nested specifications culminating in the divergence model.

    (1) Baseline.
    (2) Full text model.
    (3) Text model + Divergence.
    """
    if df is None:
        df = load_text_panel()
    specs = [
        list(FIN_VARS),
        list(FIN_VARS) + list(TEXT_VARS),
        list(FIN_VARS) + list(TEXT_VARS) + ["divergence"],
    ]
    results = [_fit(df, x) for x in specs]
    return results, df, specs


def wald_joint_zero(res, vars_to_test: list[str]) -> tuple[float, float]:
    """Joint Wald test that the coefficients in *vars_to_test* are all zero.

    Returns (statistic, p-value).  Uses the clustered covariance matrix
    already estimated in *res*.
    """
    params = res.params.loc[vars_to_test].values
    cov = res.cov.loc[vars_to_test, vars_to_test].values
    stat = float(params @ np.linalg.solve(cov, params))
    from scipy.stats import chi2
    p = float(chi2.sf(stat, df=len(vars_to_test)))
    return stat, p


# ── Multi-horizon robustness sweep ────────────────────────────────────────

# (label, dependent variable, lagged dependent variable)
HORIZONS: list[tuple[str, str, str]] = [
    ("5d",   "vol_5d",        "lagged_vol_5d"),
    ("10d",  "vol_10d",       "lagged_vol_10d"),
    ("30d",  "vol_30d",       "lagged_vol_30d"),
    ("90d",  "vol_90d",       "lagged_vol_90d"),
    ("180d", "vol_180d",      "lagged_vol_180d"),
    ("365d", "vol_next_year", "lagged_vol"),
]


def load_horizon_panel(dep: str, lagged_dep: str) -> pd.DataFrame | None:
    """Load the text panel and prune to rows usable for one horizon.

    Substitutes ``lagged_dep`` for ``lagged_vol`` so each horizon uses
    its own AR(1) control.  Returns ``None`` if the columns are missing
    (the short-horizon volatilities are an optional add-on supplied by
    ``build_volatility_horizons.py``).
    """
    df = pd.read_csv(DATA_DIR / "annual_panel_text.csv")
    if dep not in df.columns or lagged_dep not in df.columns:
        return None
    df["sic2"] = (df["sic"] // 100).astype(int)
    needed = [
        dep, lagged_dep, "log_total_assets", "leverage", "roa", "asset_growth",
        "delta_sentiment", "delta_risk", "textsim", "divergence",
        "sic2", "fiscal_year", "ticker",
    ]
    df = df[needed].dropna().copy()
    return df


def run_horizons():
    """Estimate the divergence specification on each horizon.

    Yields ``(label, results, df, fin_vars)`` so the caller can reuse the
    fitted models for printing and table formatting.  The financial-control
    block substitutes the horizon-specific lag for ``lagged_vol``.
    """
    out = []
    for label, dep, lagged_dep in HORIZONS:
        df = load_horizon_panel(dep, lagged_dep)
        if df is None or df.empty:
            print(f"  [skip] horizon {label}: columns not in panel")
            continue
        fin_vars = [lagged_dep, "log_total_assets", "leverage",
                    "roa", "asset_growth"]
        xvars = fin_vars + list(TEXT_VARS) + ["divergence"]
        res = _fit(df, xvars, dep=dep)
        out.append((label, res, df, fin_vars))
    return out


def horizons_to_latex(runs, path: Path | None = None) -> str:
    """Compact comparison table: one column per horizon.

    Reports the lagged-DV AR(1) coefficient, each text variable, and the
    divergence variable, plus N, firms, and Adj R² in the footer.
    """
    if not runs:
        return ""

    # Display variables (in this order); lagged-DV row uses a generic label.
    display = [("LAGGED", "Lagged Volatility")] + [
        (v, LABELS.get(v, v)) for v in list(TEXT_VARS) + ["divergence"]
    ]
    n_cols = len(runs)

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Post-Filing Volatility at Alternative Horizons}")
    lines.append(r"\label{tab:horizons}")
    lines.append(r"\begin{tabular}{l" + "c" * n_cols + "}")
    lines.append(r"\toprule")
    header = " & ".join(f"({i+1}) {label}"
                        for i, (label, _, _, _) in enumerate(runs))
    lines.append(f" & {header} \\\\")
    lines.append(r"\midrule")

    for var, label in display:
        coefs, tstats = [], []
        for _, res, _, fin_vars in runs:
            v = fin_vars[0] if var == "LAGGED" else var
            if v not in res.params.index:
                coefs.append("")
                tstats.append("")
                continue
            c = res.params[v]
            t = res.tstats[v]
            p = res.pvalues[v]
            coefs.append(f"${c:+.4f}${_stars(p)}")
            tstats.append(f"$({t:+.2f})$")
        lines.append(f"{label} & " + " & ".join(coefs) + r" \\")
        lines.append(" & " + " & ".join(tstats) + r" \\[4pt]")

    lines.append(r"\midrule")
    lines.append("Industry FE & " + " & ".join(["Yes"] * n_cols) + r" \\")
    lines.append("Year FE & " + " & ".join(["Yes"] * n_cols) + r" \\")
    obs = " & ".join(f"{int(res.nobs):,}" for _, res, _, _ in runs)
    firms = " & ".join(f"{df['ticker'].nunique()}" for _, _, df, _ in runs)
    adj_r2 = " & ".join(f"{res.rsquared_adj:.3f}" for _, res, _, _ in runs)
    lines.append("Observations & " + obs + r" \\")
    lines.append("Firms & " + firms + r" \\")
    lines.append(r"Adj.\ $R^2$ & " + adj_r2 + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    lines.append(r"\begin{tablenotes}")
    lines.append(
        r"\item \textit{Note:} "
        r"Each column re-estimates the divergence specification "
        r"(equation~\ref{eq:divergence_model}) with the dependent variable "
        r"replaced by the annualised standard deviation of daily log returns "
        r"over a shorter post-filing window. The lagged dependent variable "
        r"is correspondingly the same firm's volatility measured over the "
        r"prior filing year's window of the same length. "
        r"All specifications include two-digit SIC industry and fiscal-year "
        r"fixed effects; $t$-statistics in parentheses use firm-clustered "
        r"standard errors. ***, **, and * denote significance at the 1\%, "
        r"5\%, and 10\% levels, respectively."
    )
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    if path is not None:
        path.write_text(tex, encoding="utf-8")
        print(f"  Table saved → {path.name}")
    return tex


# ── LaTeX table formatting ────────────────────────────────────────────────

def _stars(pval: float) -> str:
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.10:
        return "*"
    return ""


def _regression_to_latex(results, df, specs, caption: str, label: str,
                         note: str, path: Path | None = None) -> str:
    """Generic three-column nested-regression table writer."""
    n_cols = len(results)
    all_vars = specs[-1]  # full variable set from the widest model
    n_firms = df["ticker"].nunique()
    n_industries = df["sic2"].nunique()

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

    # Coefficients
    for var in all_vars:
        label = LABELS.get(var, var)
        coefs, tstats = [], []
        for res, xvars in zip(results, specs):
            if var in xvars:
                c = res.params[var]
                t = res.tstats[var]
                p = res.pvalues[var]
                coefs.append(f"${c:.4f}${_stars(p)}")
                tstats.append(f"$({t:.2f})$")
            else:
                coefs.append("")
                tstats.append("")
        lines.append(f"{label} & " + " & ".join(coefs) + r" \\")
        lines.append(" & " + " & ".join(tstats) + r" \\[4pt]")

    # Footer
    lines.append(r"\midrule")
    lines.append("Industry FE & " + " & ".join(["Yes"] * n_cols) + r" \\")
    lines.append("Year FE & " + " & ".join(["Yes"] * n_cols) + r" \\")

    obs = [f"{res.nobs:,.0f}" for res in results]
    firms = [str(n_firms)] * n_cols
    adj_r2 = [f"{res.rsquared_adj:.3f}" for res in results]

    lines.append("Observations & " + " & ".join(obs) + r" \\")
    lines.append("Firms & " + " & ".join(firms) + r" \\")
    lines.append(r"Adj.\ $R^2$ & " + " & ".join(adj_r2) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Table note
    lines.append(r"\begin{tablenotes}")
    lines.append(
        r"\item \textit{Note:} " + note + " "
        f"All specifications include two-digit SIC industry "
        f"({n_industries} groups) and fiscal-year fixed effects. "
        r"$t$-statistics, reported in parentheses, are based on standard "
        r"errors clustered at the firm level. "
        r"***, **, and * denote significance at the 1\%, 5\%, "
        r"and 10\% levels, respectively."
    )
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)

    if path is not None:
        path.write_text(tex, encoding="utf-8")
        print(f"  Table saved → {path.name}")

    return tex


_BASELINE_NOTE = (
    r"This table reports OLS estimates of post-filing annualised "
    r"volatility on financial determinants. The dependent variable is "
    r"the annualised standard deviation of daily log returns over the "
    r"365-day window following each 10-K filing date."
)

_TEXT_NOTE = (
    r"This table reports OLS estimates of post-filing annualised "
    r"volatility on financial controls and textual features extracted "
    r"from the 10-K narrative. $\Delta$Sentiment and $\Delta$Risk are "
    r"within-firm year-on-year changes in MD\&A net tone and Item~1A "
    r"risk-word share, respectively, computed with the Loughran--McDonald "
    r"dictionary. TextSim is the cosine similarity of the firm's TF--IDF "
    r"vector with its prior-year filing."
)

_DIVERGENCE_NOTE = (
    r"This table reports OLS estimates of post-filing annualised "
    r"volatility on financial controls, textual features, and the "
    r"Divergence variable defined as $\Delta$Sentiment minus "
    r"$\lambda \cdot \Delta$ROA, where $\lambda$ is calibrated by a "
    r"within-firm OLS of $\Delta$Sentiment on $\Delta$ROA."
)


def baseline_to_latex(results, df, specs, path: Path | None = None) -> str:
    """Baseline volatility-determinants table."""
    return _regression_to_latex(
        results, df, specs,
        caption="Baseline Volatility Determinants",
        label="tab:baseline",
        note=_BASELINE_NOTE,
        path=path,
    )


def text_to_latex(results, df, specs, path: Path | None = None) -> str:
    """Text-model regression table (H1)."""
    return _regression_to_latex(
        results, df, specs,
        caption="Textual Features and Post-Filing Volatility",
        label="tab:text_regression",
        note=_TEXT_NOTE,
        path=path,
    )


def divergence_to_latex(results, df, specs, path: Path | None = None) -> str:
    """Divergence-model regression table (H2)."""
    return _regression_to_latex(
        results, df, specs,
        caption="Divergence and Post-Filing Volatility",
        label="tab:divergence_regression",
        note=_DIVERGENCE_NOTE,
        path=path,
    )


# ── Descriptive statistics ─────────────────────────────────────────────────

def descriptive_stats_to_latex(df: pd.DataFrame,
                               path: Path | None = None) -> str:
    """Generate Panel A: Descriptive Statistics."""
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
    n_firms = df["ticker"].nunique()
    lines.append(
        rf"Observations & \multicolumn{{5}}{{c}}{{{len(df):,}}} \\"
    )
    lines.append(
        rf"Firms & \multicolumn{{5}}{{c}}{{{n_firms}}} \\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(
        r"\item \textit{Note:} "
        r"This table reports summary statistics for the regression sample. "
        r"Volatility$_{t+1}$ is the annualised standard deviation of daily "
        r"log returns over the 365-day window following each 10-K filing date. "
        r"Leverage, ROA, and Asset Growth are winsorised at the "
        r"1st and 99th percentiles."
    )
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    if path is not None:
        path.write_text(tex, encoding="utf-8")
        print(f"  Table saved → {path.name}")
    return tex


# ── Correlation matrix ────────────────────────────────────────────────────

def correlation_to_latex(df: pd.DataFrame,
                         path: Path | None = None) -> str:
    """Generate Pearson correlation matrix with significance stars."""
    from scipy.stats import pearsonr

    n = len(PANEL_VARS)
    corr_vals = pd.DataFrame(np.nan, index=PANEL_VARS, columns=PANEL_VARS)
    pvals = pd.DataFrame(np.nan, index=PANEL_VARS, columns=PANEL_VARS)

    for i in range(n):
        for j in range(n):
            r, p = pearsonr(df[PANEL_VARS[i]], df[PANEL_VARS[j]])
            corr_vals.iloc[i, j] = r
            pvals.iloc[i, j] = p

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Pearson Correlation Matrix}")
    lines.append(r"\label{tab:correlation}")
    col_spec = "l" + "c" * n
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Column numbers header
    header = " & ".join([f"({i+1})" for i in range(n)])
    lines.append(f" & {header} \\\\")
    lines.append(r"\midrule")

    # Lower triangle only (standard in finance)
    for i in range(n):
        label = LABELS.get(PANEL_VARS[i], PANEL_VARS[i])
        cells = []
        for j in range(n):
            if j > i:
                cells.append("")
            elif i == j:
                cells.append("1")
            else:
                r = corr_vals.iloc[i, j]
                p = pvals.iloc[i, j]
                cells.append(f"{r:.2f}{_stars(p)}")
        row_label = f"({i+1}) {label}"
        lines.append(f"{row_label} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\begin{tablenotes}")
    lines.append(
        r"\item \textit{Note:} "
        r"This table reports pairwise Pearson correlation coefficients. "
        r"***, **, and * denote significance at the 1\%, 5\%, "
        r"and 10\% levels, respectively."
    )
    lines.append(r"\end{tablenotes}")
    lines.append(r"\end{table}")

    tex = "\n".join(lines)
    if path is not None:
        path.write_text(tex, encoding="utf-8")
        print(f"  Table saved → {path.name}")
    return tex


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Loading panel …")
    df = load_regression_panel()
    print(f"  {df['ticker'].nunique()} firms, {len(df):,} observations, "
          f"FY {df['fiscal_year'].min()}–{df['fiscal_year'].max()}")

    print("\nGenerating summary statistics …")
    descriptive_stats_to_latex(df, TEX_TABLE_DIR / "summary_statistics.tex")

    print("Generating correlation matrix …")
    correlation_to_latex(df, TEX_TABLE_DIR / "correlation_matrix.tex")

    print("\nRunning baseline regressions …")
    results, df, specs = run_baseline(df)

    for i, res in enumerate(results):
        print(f"\n{'='*60}")
        print(f"  Model ({i+1}): {', '.join(specs[i])}")
        print(f"{'='*60}")
        print(f"  N = {res.nobs:,.0f}  |  R² = {res.rsquared:.4f}  |  "
              f"Adj R² = {res.rsquared_adj:.4f}")
        for var in specs[i]:
            c = res.params[var]
            t = res.tstats[var]
            p = res.pvalues[var]
            print(f"  {var:20s}  β = {c:+.4f}  t = {t:+.2f}  p = {p:.4f} {_stars(p)}")

    tex_path = TEX_TABLE_DIR / "baseline_regression.tex"
    baseline_to_latex(results, df, specs, path=tex_path)

    # ── Text + divergence models ────────────────────────────────────────
    text_panel_path = DATA_DIR / "annual_panel_text.csv"
    if not text_panel_path.exists():
        print(f"\n[skip] {text_panel_path.name} not found — "
              "run build_text_features.py to enable text/divergence models.")
        print("\nDone.")
        return

    print("\nLoading text-extended panel …")
    tdf = load_text_panel()
    print(f"  {tdf['ticker'].nunique()} firms, {len(tdf):,} observations")

    print("\nRunning text models (H1) …")
    t_results, tdf, t_specs = run_text(tdf)
    for i, res in enumerate(t_results):
        print(f"  Model ({i+1}): N={res.nobs:,.0f}  Adj R²={res.rsquared_adj:.4f}")
    text_to_latex(t_results, tdf, t_specs,
                  path=TEX_TABLE_DIR / "text_regression.tex")

    stat, pval = wald_joint_zero(t_results[-1], TEXT_VARS)
    print(f"  H1 Wald χ²({len(TEXT_VARS)}) = {stat:.2f}, p = {pval:.4g}")

    print("\nRunning divergence models (H2) …")
    d_results, _, d_specs = run_divergence(tdf)
    for i, res in enumerate(d_results):
        print(f"  Model ({i+1}): N={res.nobs:,.0f}  Adj R²={res.rsquared_adj:.4f}")
    divergence_to_latex(d_results, tdf, d_specs,
                        path=TEX_TABLE_DIR / "divergence_regression.tex")
    div_res = d_results[-1]
    c = div_res.params["divergence"]
    t = div_res.tstats["divergence"]
    p = div_res.pvalues["divergence"]
    print(f"  H2 δ_divergence = {c:+.4f}  t = {t:+.2f}  p = {p:.4g}")

    # ── Robustness: alternative post-filing horizons ────────────────────
    print("\nRunning multi-horizon robustness sweep …")
    runs = run_horizons()
    if runs:
        for label, res, _df_h, _ in runs:
            d = res.params["divergence"]
            td = res.tstats["divergence"]
            pd_ = res.pvalues["divergence"]
            print(f"  {label:>5s}: N={int(res.nobs):,}  "
                  f"Adj R²={res.rsquared_adj:.4f}  "
                  f"δ_div={d:+.4f}  t={td:+.2f}  p={pd_:.4g}")
        horizons_to_latex(runs, path=TEX_TABLE_DIR / "horizons_regression.tex")
    else:
        print("  [skip] no horizon columns in panel — "
              "run build_volatility_horizons.py")

    print("\nDone.")


if __name__ == "__main__":
    main()

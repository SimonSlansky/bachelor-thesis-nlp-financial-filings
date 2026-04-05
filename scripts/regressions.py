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
    "lagged_vol": "Lagged Volatility",
    "log_total_assets": r"$\ln(\text{Total Assets})$",
    "leverage": "Leverage",
    "roa": "ROA",
    "asset_growth": "Asset Growth",
}


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


# ── LaTeX table formatting ────────────────────────────────────────────────

def _stars(pval: float) -> str:
    if pval < 0.01:
        return "***"
    if pval < 0.05:
        return "**"
    if pval < 0.10:
        return "*"
    return ""


def baseline_to_latex(results, df, specs, path: Path | None = None) -> str:
    """Format baseline results as a publication-quality LaTeX table."""
    n_cols = len(results)
    all_vars = specs[-1]  # full variable set from the widest model
    n_firms = df["ticker"].nunique()
    n_industries = df["sic2"].nunique()

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Baseline Volatility Determinants}")
    lines.append(r"\label{tab:baseline}")
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
                coefs.append(f"${c:+.4f}${_stars(p)}")
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
    r2 = [f"{res.rsquared:.3f}" for res in results]
    adj_r2 = [f"{res.rsquared_adj:.3f}" for res in results]

    lines.append("Observations & " + " & ".join(obs) + r" \\")
    lines.append("Firms & " + " & ".join(firms) + r" \\")
    lines.append(r"Adj.\ $R^2$ & " + " & ".join(adj_r2) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    # Table note
    lines.append(r"\begin{tablenotes}")
    lines.append(
        r"\item \textit{Note:} "
        r"This table reports OLS estimates of post-filing annualised "
        r"volatility on financial determinants. The dependent variable is "
        r"the annualised standard deviation of daily log returns over the "
        r"365-day window following each 10-K filing date. "
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


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    print("Loading panel …")
    df = load_regression_panel()
    print(f"  {df['ticker'].nunique()} firms, {len(df):,} observations, "
          f"FY {df['fiscal_year'].min()}–{df['fiscal_year'].max()}")

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
    print(f"\nDone.")


if __name__ == "__main__":
    main()

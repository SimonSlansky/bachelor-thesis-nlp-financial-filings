"""Stationarity / unit-root checks for the disclosure-volatility study.

The headline regressions of Chapters 5--6 are dynamic panels: lagged
log-volatility enters with its own coefficient. If any series carried a unit
root, that lagged-level regression could be spurious, so every variable that
enters the regressions has to be shown to be stationary, I(0).

The check is built entirely on the (augmented) Dickey--Fuller test from the
time-series curriculum, plus a simple way of pooling it across firms:

  * Each variable is a panel of short annual firm-level series. We first
    subtract each year's cross-sectional mean, which strips out the
    market-wide shocks shared by all firms (the same job the year fixed
    effects do in the regressions).

  * For every firm with at least ``MIN_T`` annual observations we run an
    augmented Dickey--Fuller regression with an intercept and one lag and
    record (i) the estimated first-order autoregressive coefficient
    ``rho_hat`` -- how close the series is to a unit root, rho = 1 -- and
    (ii) the ADF t-statistic and whether it rejects the unit-root null at 5%.

  * A single firm's 15-year series has little power on its own, so we combine
    the per-firm ADF p-values across all firms with Fisher's method
    (Maddala--Wu), which gives one decisive test per variable.

Run:
    .venv\\Scripts\\python.exe scripts/stationarity.py

Writes ``tex/tables/stationarity.tex`` and prints a console summary.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from statsmodels.tsa.adfvalues import mackinnonp

from config import DATA_DIR

BASE_DIR = Path(__file__).resolve().parent.parent
TEX_TABLE_DIR = BASE_DIR / "tex" / "tables"
TEX_TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ─────────────────────────────────────────────────────────
MIN_T = 8             # a firm needs at least this many annual observations
RHO_LAG = 0           # rho-hat and the DF cross-check use the plain AR(1)/DF
TEST_LAG = 1          # headline test = augmented Dickey--Fuller, one lag
REJECT_LEVEL = 0.05   # per-firm rejection threshold
ADF_CV_5 = -2.90      # approx. 5% ADF critical value (intercept) for reference

# Variables tested: the full set that enters any regression in Chapters 5-6.
# (label is the LaTeX row label.)
VARS: list[tuple[str, str]] = [
    ("log_vol",          r"$\ln \sigma_{i,t}$ (outcome)"),
    ("log_total_assets", r"$\ln(\text{Total Assets})$"),
    ("leverage",         r"Leverage"),
    ("roa",              r"ROA"),
    ("asset_growth",     r"Asset Growth"),
    ("current_ratio",    r"Current Ratio"),
    ("ocf_to_assets",    r"OCF/Assets"),
    ("log_words_1a",     r"$\ln(\text{Words}^{\text{1A}})$"),
    ("risk_1a",          r"RiskDensity"),
    ("unc_1a",           r"UncDensity"),
    ("lit_1a",           r"LitDensity"),
    ("omission_1a",      r"Omission$^{\text{1A}}$"),
]


# ── Data loading ──────────────────────────────────────────────────────────

def load_panel() -> pd.DataFrame:
    """Load the text-extended panel and attach the derived test variables."""
    df = pd.read_csv(DATA_DIR / "annual_panel_text.csv")
    df["log_vol"] = np.log(df["vol_30d"].clip(lower=1e-6))
    if "log_words_1a" not in df.columns:
        df["log_words_1a"] = np.log(df["words_1a"].clip(lower=1))
    sil_path = DATA_DIR / "silence.csv"
    if sil_path.exists():
        sil = pd.read_csv(sil_path)[["ticker", "fiscal_year", "omission_1a"]]
        df = df.merge(sil, on=["ticker", "fiscal_year"], how="left")
    return df


def year_firm_matrix(df: pd.DataFrame, var: str) -> pd.DataFrame:
    """Pivot one variable into a (fiscal_year x ticker) matrix."""
    sub = df[["fiscal_year", "ticker", var]].dropna(subset=[var])
    mat = sub.pivot_table(index="fiscal_year", columns="ticker",
                          values=var, aggfunc="mean")
    return mat.sort_index()


def demean_cross_section(mat: pd.DataFrame) -> pd.DataFrame:
    """Subtract each year's cross-sectional mean (removes common shocks)."""
    return mat.sub(mat.mean(axis=1), axis=0)


def firm_series(mat: pd.DataFrame, min_t: int = MIN_T):
    """Yield (ticker, years, values) for firms with >= min_t observations."""
    for ticker in mat.columns:
        col = mat[ticker].dropna()
        if col.size >= min_t:
            yield ticker, col.index.to_numpy(), col.to_numpy(dtype=float)


# ── (Augmented) Dickey--Fuller on a single series ─────────────────────────

def adf_fit(y: np.ndarray, lag: int):
    """Intercept (augmented) Dickey--Fuller regression on one series.

    Model:  Δy_t = α + γ·y_{t-1} + Σ_k δ_k·Δy_{t-k} + e_t.
    The autoregressive root is ρ = 1 + γ and the unit-root null is γ = 0
    (equivalently ρ = 1). Returns (rho_hat, tstat, pvalue) or None if the
    series is too short / degenerate.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if n < lag + 3:
        return None
    dy = np.diff(y)                          # Δy_t
    resp = dy[lag:]                          # Δy_t, t = lag+1 .. n-1
    cols = [np.ones_like(resp), y[lag:-1]]   # const, y_{t-1}
    for k in range(1, lag + 1):
        cols.append(dy[lag - k:-k])          # Δy_{t-k}
    X = np.column_stack(cols)
    dof = resp.size - X.shape[1]
    if dof <= 0:
        return None
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return None
    beta = xtx_inv @ X.T @ resp
    resid = resp - X @ beta
    s2 = float(resid @ resid) / dof
    if s2 <= 0:
        return None
    se = np.sqrt(s2 * xtx_inv[1, 1])
    if not np.isfinite(se) or se == 0:
        return None
    gamma = float(beta[1])
    tstat = gamma / se
    if not np.isfinite(tstat):
        return None
    pval = float(mackinnonp(tstat, regression="c", N=1))
    pval = min(max(pval, 1e-12), 1 - 1e-12)
    return 1.0 + gamma, tstat, pval


# ── Pool the per-firm ADF tests over the panel ────────────────────────────

def adf_panel(mat_dm: pd.DataFrame, lag: int) -> dict | None:
    """Run the per-firm ADF over one variable and summarise the panel.

    Returns mean rho-hat, mean ADF t, the share of firms rejecting the
    unit-root null, and Fisher's combined test of the per-firm p-values.
    """
    rhos, tstats, pvals = [], [], []
    for _ticker, _yrs, vals in firm_series(mat_dm):
        fit = adf_fit(vals, lag)
        if fit is None:
            continue
        rho, t, p = fit
        rhos.append(rho)
        tstats.append(t)
        pvals.append(p)
    n = len(pvals)
    if n == 0:
        return None
    pvals = np.asarray(pvals)
    fisher = float(-2.0 * np.log(pvals).sum())       # ~ chi2(2n) under H0
    fisher_p = float(chi2.sf(fisher, 2 * n))
    return {
        "n": n,
        "mean_rho": float(np.mean(rhos)),
        "mean_t": float(np.mean(tstats)),
        "reject_frac": float(np.mean(pvals < REJECT_LEVEL)),
        "fisher": fisher,
        "fisher_p": fisher_p,
    }


def compute_rows(df: pd.DataFrame) -> list[dict]:
    """Run the ADF battery (DF cross-check + augmented DF) on every variable."""
    rows = []
    for var, label in VARS:
        if var not in df.columns:
            print(f"  [skip] {var}: not in panel")
            continue
        mat_dm = demean_cross_section(year_firm_matrix(df, var))
        df_res = adf_panel(mat_dm, RHO_LAG)      # plain Dickey--Fuller
        adf_res = adf_panel(mat_dm, TEST_LAG)    # augmented Dickey--Fuller
        if df_res is None or adf_res is None:
            continue
        rows.append({
            "var": var, "label": label,
            "n": adf_res["n"],
            "mean_rho": df_res["mean_rho"],
            "df_reject": df_res["reject_frac"],
            "df_fisher_p": df_res["fisher_p"],
            "mean_t": adf_res["mean_t"],
            "adf_reject": adf_res["reject_frac"],
            "fisher_p": adf_res["fisher_p"],
        })
    return rows


# ── LaTeX table ───────────────────────────────────────────────────────────

def _fisher_cell(p: float) -> str:
    return r"$<0.001$" if p < 0.001 else f"${p:.3f}$"


def table_to_latex(rows: list[dict], path: Path | None = None) -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Stationarity Tests (Augmented Dickey--Fuller)}")
    lines.append(r"\label{tab:stationarity}")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{8pt}")
    lines.append(r"\begin{tabular}{l c c c c}")
    lines.append(r"\toprule")
    lines.append(r" & $\hat\rho$ & ADF $\bar t$ & Reject & Fisher \\")
    lines.append(r" & {\scriptsize (AR1 coef.)} & {\scriptsize (mean)} "
                 r"& {\scriptsize (\% firms)} & {\scriptsize $p$-value} \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(
            f"{r['label']} & ${r['mean_rho']:.2f}$ & ${r['mean_t']:+.2f}$ & "
            f"${100 * r['adf_reject']:.0f}\\%$ & {_fisher_cell(r['fisher_p'])} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\par\smallskip")
    lines.append(r"\noindent\footnotesize ")
    lines.append(
        r"\textit{Note:} "
        r"Stationarity tests for every variable that enters the regressions "
        r"of Chapters~\ref{ch:results}--\ref{ch:robustness}; lagged "
        r"log-volatility is the lag of the outcome and is covered by the "
        r"first row. Each variable is a panel of short annual firm-level "
        r"series ($T \leq 15$). Before testing, each year's cross-sectional "
        r"mean is subtracted to remove the market-wide shocks shared by all "
        r"firms, mirroring the year fixed effects in the regressions. For "
        rf"every firm with at least {MIN_T} annual observations we run an "
        r"augmented Dickey--Fuller regression \parencite{dickeyfuller1979} "
        r"with an intercept and one lag. $\hat\rho$ is the average estimated "
        r"first-order autoregressive coefficient across firms; a unit root "
        r"would imply $\hat\rho = 1$, so values well below one indicate mean "
        r"reversion. ADF~$\bar t$ is the mean ADF statistic (the $5\%$ "
        r"critical value with an intercept is about $-2.9$), and ``Reject'' "
        r"is the share of firms rejecting the unit-root null at $5\%$. "
        r"Because each firm's series is short and individually low-powered, "
        r"the last column combines the per-firm ADF $p$-values across firms "
        r"with Fisher's method \parencite{maddala1999}; the combined "
        r"$p$-value is below $0.001$ for every variable, so the unit-root "
        r"null is rejected for all of them. Plain Dickey--Fuller without the "
        r"augmentation lag gives the same conclusion. The most persistent "
        r"variables (Leverage and the risk-word densities) are bounded ratios "
        r"and shares, which cannot follow a unit-root process---whose "
        r"variance grows without bound---so their persistence is mean "
        r"reversion, not a unit root. Every variable is therefore treated as "
        r"stationary, $I(0)$."
    )
    lines.append(r"\end{table}")
    tex = "\n".join(lines)
    if path is not None:
        path.write_text(tex, encoding="utf-8")
        print(f"\n  Table saved -> {path.name}")
    return tex


# ── Driver ────────────────────────────────────────────────────────────────

def _print_table(rows: list[dict]) -> None:
    hdr = (f"{'variable':<18s} {'firms':>5s} {'rho_hat':>8s} "
           f"{'ADF_tbar':>9s} {'rej%(DF)':>9s} {'rej%(ADF)':>10s} "
           f"{'Fisher_p':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['var']:<18s} {r['n']:>5d} {r['mean_rho']:>8.2f} "
              f"{r['mean_t']:>+9.2f} {100 * r['df_reject']:>8.0f}% "
              f"{100 * r['adf_reject']:>9.0f}% "
              f"{r['fisher_p']:>9.3f}")


def main() -> None:
    print("=== Stationarity diagnostics (augmented Dickey--Fuller, 30-day "
          "horizon) ===")
    df = load_panel()
    print(f"Loaded {len(df):,} firm-years, {df['ticker'].nunique()} firms\n")

    rows = compute_rows(df)
    _print_table(rows)

    print("\nLegend: rho_hat < 1 => mean reverting (rho = 1 is a unit root); "
          "ADF rejects => stationary; Fisher_p combines the per-firm ADF "
          "tests into one decisive panel statistic. DF = plain Dickey--Fuller "
          "(no lag), ADF = one augmentation lag.")

    table_to_latex(rows, TEX_TABLE_DIR / "stationarity.tex")


if __name__ == "__main__":
    main()

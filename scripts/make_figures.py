"""Generate the thesis figures (PDF) consumed by the LaTeX chapters.

Figures produced (written to ``tex/img/``):

  fig_volatility_time.pdf   Figure 2 — median post-filing 30-day volatility
                            by fiscal year, with the inter-quartile band.
                            Motivates the year fixed effects and the
                            COVID-era sub-sample split.

  fig_coef_fe.pdf           Figure 6 — H1/H2 point estimates with 95%
                            confidence intervals under industry+year FE
                            versus firm+year FE. Shows that the density
                            effects are a between-firm regularity.

  fig_horizon_profile.pdf   Figure 7 — the length (H1) and litigious-density
                            (H2) coefficients with 95% CIs across the six
                            post-filing horizons (5-365 days).

  fig_omission_length.pdf   Figure 9 — scatter of the Item 1A omission gap
                            against Item 1A length, with the OLS fit line.
                            Communicates the r = -0.85 correlation that
                            motivates the H3 robustness checks.

The figures reuse the exact estimation code in ``regressions.py`` so that
the numbers match the tables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_DIR
from regressions import (
    DENS_VAR,
    LEN_VAR,
    LIT_VAR,
    UNC_VAR,
    HORIZONS,
    load_text_panel,
    _fit,
    run_firm_fe,
)

BASE_DIR = Path(__file__).resolve().parent.parent
IMG_DIR = BASE_DIR / "tex" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)

# A restrained, print-friendly style consistent with a black-and-white thesis.
plt.rcParams.update({
    "font.size": 10,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "figure.dpi": 150,
})

Z95 = 1.959964  # two-sided 95% normal critical value


def _save(fig, name: str) -> None:
    path = IMG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved -> {path.name}")


# ── Figure 2: volatility over time ────────────────────────────────────────

def fig_volatility_time() -> None:
    """Median annualised 30-day post-filing volatility by fiscal year."""
    df = pd.read_csv(DATA_DIR / "annual_panel_text.csv")
    df = df[["fiscal_year", "vol_30d"]].dropna()
    g = df.groupby("fiscal_year")["vol_30d"]
    stats = pd.DataFrame({
        "p25": g.quantile(0.25),
        "median": g.median(),
        "p75": g.quantile(0.75),
    }).reset_index()

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.fill_between(stats["fiscal_year"], stats["p25"], stats["p75"],
                    color="0.80", label="Inter-quartile range")
    ax.plot(stats["fiscal_year"], stats["median"],
            color="black", linewidth=1.6, marker="o", markersize=3.5,
            label="Median")
    ax.set_xlabel("Fiscal year")
    ax.set_ylabel(r"Annualised 30-day volatility $\sigma_{i,t+1}$")
    ax.set_xticks(stats["fiscal_year"][::2])
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor="white",
        edgecolor="0.85",
        fontsize=9,
        borderpad=0.5,
        handlelength=1.8,
    )
    _save(fig, "fig_volatility_time.pdf")


# ── Figure 6: coefficient plot, industry FE vs firm FE ────────────────────

def fig_coef_fe() -> None:
    """H1/H2 coefficients with 95% CIs under industry FE vs firm FE."""
    # Industry + year FE, H1 + LM-list decomposition, 30-day horizon.
    df_ind = load_text_panel(vol_col="vol_30d", lag_col="lagged_vol_30d")
    res_ind = _fit(df_ind, ["lagged_log_vol",
                            "log_total_assets", "leverage", "roa",
                            "asset_growth", LEN_VAR, UNC_VAR, LIT_VAR])
    # Firm + year FE, same specification.
    res_ffe, _ = run_firm_fe(vol_col="vol_30d", lag_col="lagged_vol_30d")

    terms = [
        (LEN_VAR, r"$\ln(\mathrm{Words}^{1A})$" + "\n(H1 length)"),
        (UNC_VAR, "UncDensity\n(H2)"),
        (LIT_VAR, "LitDensity\n(H2)"),
    ]

    def grab(res):
        b = [res.params[v] for v, _ in terms]
        se = [res.std_errors[v] for v, _ in terms]
        return np.array(b), np.array(se)

    b_ind, se_ind = grab(res_ind)
    b_ffe, se_ffe = grab(res_ffe)

    y = np.arange(len(terms))[::-1]  # top-to-bottom order
    off = 0.14

    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.axvline(0.0, color="0.6", linewidth=0.8, linestyle="--")
    ax.errorbar(b_ind, y + off, xerr=Z95 * se_ind, fmt="o",
                color="black", markersize=4, capsize=3, linewidth=1.2,
                label="Industry + year FE")
    ax.errorbar(b_ffe, y - off, xerr=Z95 * se_ffe, fmt="s",
                color="0.5", markersize=4, capsize=3, linewidth=1.2,
                label="Firm + year FE")
    ax.set_yticks(y)
    ax.set_yticklabels([lbl for _, lbl in terms])
    ax.set_xlabel("Coefficient on $\\ln(\\sigma_{i,t+1})$ (95% CI)")
    ax.legend(frameon=False, loc="lower right")
    _save(fig, "fig_coef_fe.pdf")


# ── Figure 7: horizon profile ─────────────────────────────────────────────

def fig_horizon_profile() -> None:
    """Length (H1) and litigious-density (H2) coefficients across horizons."""
    h_days = {"5d": 5, "10d": 10, "30d": 30,
              "90d": 90, "180d": 180, "365d": 365}
    rows = []
    for label, vol_col, lag_col in HORIZONS:
        df = load_text_panel(vol_col=vol_col, lag_col=lag_col)
        res = _fit(df, ["lagged_log_vol",
                        "log_total_assets", "leverage", "roa",
                        "asset_growth", LEN_VAR, UNC_VAR, LIT_VAR])
        rows.append({
            "h": h_days[label],
            "b_len": res.params[LEN_VAR],
            "se_len": res.std_errors[LEN_VAR],
            "b_lit": res.params[LIT_VAR],
            "se_lit": res.std_errors[LIT_VAR],
        })
    d = pd.DataFrame(rows).sort_values("h")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.6, 3.2))

    ax1.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--")
    ax1.errorbar(d["h"], d["b_len"], yerr=Z95 * d["se_len"], fmt="o-",
                 color="black", markersize=4, capsize=3, linewidth=1.2)
    ax1.set_title("H1: Item 1A length", fontsize=10)
    ax1.set_xlabel("Horizon $h$ (days)")
    ax1.set_ylabel("Coefficient (95% CI)")
    ax1.set_xscale("log")
    ax1.set_xticks(d["h"])
    ax1.set_xticklabels(d["h"].astype(int))
    ax1.minorticks_off()

    ax2.axhline(0.0, color="0.6", linewidth=0.8, linestyle="--")
    ax2.errorbar(d["h"], d["b_lit"], yerr=Z95 * d["se_lit"], fmt="s-",
                 color="0.4", markersize=4, capsize=3, linewidth=1.2)
    ax2.set_title("H2: LM-Litigious density", fontsize=10)
    ax2.set_xlabel("Horizon $h$ (days)")
    ax2.set_xscale("log")
    ax2.set_xticks(d["h"])
    ax2.set_xticklabels(d["h"].astype(int))
    ax2.minorticks_off()

    fig.tight_layout()
    _save(fig, "fig_horizon_profile.pdf")


# ── Figure 9: omission gap vs Item 1A length ──────────────────────────────

def fig_omission_length() -> None:
    """Scatter of the Item 1A omission gap against Item 1A length.

    Firm-years with non-recoverable Item 1A text (``words_1a == 0``) are
    excluded: they mechanically take ``omission_1a = 1`` and
    ``log_words_1a = 0`` because the extraction returned an empty section,
    not because of disclosure behaviour. They never enter any regression
    (the LM-density variables are undefined for them), so this filter keeps
    the visual diagnostic consistent with the estimation sample.
    """
    sil = pd.read_csv(DATA_DIR / "silence.csv")[
        ["ticker", "fiscal_year", "omission_1a"]
    ]
    panel = pd.read_csv(DATA_DIR / "annual_panel_text.csv")
    if "log_words_1a" not in panel.columns:
        panel["log_words_1a"] = np.log(panel["words_1a"].clip(lower=1))
    panel = panel[["ticker", "fiscal_year", "log_words_1a", "words_1a"]]
    d = sil.merge(panel, on=["ticker", "fiscal_year"], how="inner").dropna()

    # Drop non-recoverable Item 1A extractions (words_1a == 0 ⇒ the
    # mechanical [0, 1] corner mass that is not a disclosure choice).
    n_before = len(d)
    d = d[d["words_1a"] > 0].copy()
    n_removed = n_before - len(d)
    print(f"  fig_omission_length: excluded {n_removed} zero-length "
          f"Item 1A firm-years (words_1a == 0)")

    x = d["log_words_1a"].to_numpy()
    y = d["omission_1a"].to_numpy()
    r = np.corrcoef(x, y)[0, 1]
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.scatter(x, y, s=6, color="0.55", alpha=0.35, edgecolors="none")
    ax.plot(xs, intercept + slope * xs, color="black", linewidth=1.8,
            label=f"OLS fit ($r = {r:.2f}$)")
    ax.set_xlabel(r"Item~1A length $\ln(\mathrm{Words}^{1A})$")
    ax.set_ylabel(r"Omission gap $\mathrm{Omission}^{1A}$")
    ax.legend(frameon=False, loc="upper right")
    _save(fig, "fig_omission_length.pdf")


def main() -> None:
    print("Building thesis figures...")
    fig_volatility_time()
    fig_coef_fe()
    fig_horizon_profile()
    fig_omission_length()
    print("Done.")


if __name__ == "__main__":
    main()

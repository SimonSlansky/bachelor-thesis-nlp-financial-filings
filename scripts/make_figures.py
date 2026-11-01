"""Thesis figures (PDF, tex/img/), read from data/results/ and the panel.

  fig_volatility_time.pdf   median 30-day post-filing volatility by fiscal year (IQR band)
  fig_coef_fe.pdf           text coefficients, industry FE vs firm FE (firm_fe.csv)
  fig_horizon_profile.pdf   length and LitDensity coefficients by horizon (horizons.csv)
  fig_omission_length.pdf   omission gap vs Item 1A length, H3 30-day sample
  fig_placebo.pdf           1,000 placebo t-statistics vs the observed one
  fig_timeline.pdf          timing of all variables around one filing (#29)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_DIR
from regressions import CONTROLS, LEN_VAR, DENS_VAR, load_text_panel

BASE_DIR = Path(__file__).resolve().parent.parent
IMG_DIR = BASE_DIR / "tex" / "img"
IMG_DIR.mkdir(parents=True, exist_ok=True)
RES = DATA_DIR / "results"

plt.rcParams.update({
    "font.size": 10, "font.family": "serif", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3,
    "grid.linewidth": 0.5, "figure.dpi": 150,
})
Z95 = 1.959964


def _save(fig, name: str) -> None:
    fig.savefig(IMG_DIR / name, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved -> {name}")


def fig_volatility_time() -> None:
    df = load_text_panel("30d")
    df["vol"] = np.exp(df["log_vol"])
    g = df.groupby("fiscal_year")["vol"]
    s = pd.DataFrame({"p25": g.quantile(.25), "median": g.median(), "p75": g.quantile(.75)}).reset_index()
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.fill_between(s["fiscal_year"], s["p25"], s["p75"], color="0.80", label="Inter-quartile range")
    ax.plot(s["fiscal_year"], s["median"], color="black", linewidth=1.6, marker="o",
            markersize=3.5, label="Median")
    ax.set_xlabel("Fiscal year")
    ax.set_ylabel(r"Annualised 30-day volatility $\sigma^{[30]}_{i,t+1}$")
    ax.set_xticks(s["fiscal_year"][::2])
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="0.85", fontsize=9)
    _save(fig, "fig_volatility_time.pdf")


def fig_coef_fe() -> None:
    m = pd.read_csv(RES / "firm_fe.csv")
    terms = [("log_words_1a", "(5)", "$\\ln(\\mathrm{Words}^{1A})$\n(H1)"),
             ("risk_1a", "(4)", "RiskDensity\n(H2)"),
             ("unc_1a", "(5)", "UncDensity\n(H2)"), ("lit_1a", "(5)", "LitDensity\n(H2)")]

    def grab(an):
        b, se = [], []
        for v, col, _ in terms:
            r = m[(m.column == f"{col}_no_ldv_{an}") & (m["var"] == v)].iloc[0]
            b.append(r.coef); se.append(r.se)
        return np.array(b), np.array(se)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2), gridspec_kw={"width_ratios": [1, 3]})
    for k, (ax, idx) in enumerate(zip(axes, ([0], [1, 2, 3]))):
        y = np.arange(len(idx))[::-1]
        for an, off, fmt, colr, lab in (("ind+year", .14, "o", "black", "Industry + year FE"),
                                        ("firm+year", -.14, "s", "0.5", "Firm + year FE")):
            b, se = grab(an)
            ax.errorbar(b[idx], y + off, xerr=Z95 * se[idx], fmt=fmt, color=colr, markersize=4,
                        capsize=3, linewidth=1.2, label=lab)
        ax.axvline(0, color="0.6", linewidth=0.8, linestyle="--")
        ax.set_yticks(y)
        ax.set_ylim(-0.6, len(idx) - 0.4)
        ax.set_yticklabels([terms[i][2] for i in idx], fontsize=8)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=2, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))
    fig.supxlabel("Coefficient, 30-day window (95% CI)", fontsize=9, y=0.04)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "fig_coef_fe.pdf")


def fig_horizon_profile() -> None:
    m = pd.read_csv(RES / "horizons.csv")
    m = m[m.column == "(5)"].copy()
    m["days"] = m["h"].str[:-1].astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.2))
    for ax, v, title, fmt in ((axes[0], "log_words_1a", "H1: Item 1A length", "o-"),
                              (axes[1], "lit_1a", "H2: LitDensity", "s-")):
        d = m[m["var"] == v].sort_values("days")
        ax.axhline(0, color="0.6", linewidth=0.8, linestyle="--")
        ax.errorbar(d["days"], d["coef"], yerr=Z95 * d["se"], fmt=fmt, color="black",
                    markersize=4, capsize=3, linewidth=1.2)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Horizon $h$ (days)")
        ax.set_xscale("log"); ax.set_xticks(d["days"]); ax.set_xticklabels(d["days"]); ax.minorticks_off()
    axes[0].set_ylabel("Coefficient (95% CI)")
    fig.tight_layout()
    _save(fig, "fig_horizon_profile.pdf")


def fig_omission_length() -> None:
    d = load_text_panel("30d", xvars=CONTROLS + [LEN_VAR, DENS_VAR, "omission_1a"])
    x, y = d[LEN_VAR].to_numpy(), d["omission_1a"].to_numpy()
    r = np.corrcoef(x, y)[0, 1]
    b1, b0 = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.scatter(x, y, s=6, color="0.55", alpha=0.35, edgecolors="none")
    ax.plot(xs, b0 + b1 * xs, color="black", linewidth=1.8, label=f"OLS fit ($r = {r:.2f}$, $n = {len(d):,}$)")
    ax.set_xlabel(r"$\ln(\mathrm{Words}^{1A})$")
    ax.set_ylabel(r"Omission$^{1A}$")
    ax.legend(frameon=False, loc="upper right")
    _save(fig, "fig_omission_length.pdf")


def fig_placebo() -> None:
    p = pd.read_csv(DATA_DIR / "silence_placebo.csv")
    s = pd.read_csv(RES / "h3_placebo.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    ax.hist(p["t"], bins=40, color="0.7", edgecolor="white")
    ax.axvline(s.t_obs, color="black", linewidth=1.6, label=f"observed $t = {s.t_obs:.2f}$")
    ax.axvline(s.t_p975, color="0.4", linewidth=1.0, linestyle="--",
               label=f"placebo 97.5th percentile ({s.t_p975:.2f})")
    ax.set_xlabel(r"$t$-statistic of Omission$^{1A}$ (firm + year FE, 30 days)")
    ax.set_ylabel("Permutations")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, "fig_placebo.pdf")


def fig_timeline() -> None:
    """Timing of the variables around one filing (#29); schematic, no data."""
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    ax.set_xlim(-4.2, 7.0)
    ax.set_ylim(-1.5, 1.6)
    ax.axis("off")
    ax.axhline(0, color="black", linewidth=1.0, xmin=0.02, xmax=0.98)
    marks = [(-2.2, "fiscal year\nend $t$"), (0.0, "10-K filing\n(day 0)"),
             (1.9, "day 0 + 2\ntrading days")]
    for x, lab in marks:
        ax.plot([x, x], [-0.12, 0.12], color="black", linewidth=1.2)
        ax.text(x, -0.22, lab, ha="center", va="top", fontsize=8)
    ax.annotate("", xy=(6.6, 0.4), xytext=(1.9, 0.4),
                arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.2))
    ax.text(4.25, 0.5, "outcome window: $h$ calendar days\n"
            r"$\ln\sigma^{[h]}_{i,t+1}$, market and idiosyncratic volatility",
            ha="center", va="bottom", fontsize=8)
    ax.annotate("", xy=(-0.05, 1.05), xytext=(-1.4, 1.05),
                arrowprops=dict(arrowstyle="<->", color="0.4", linewidth=1.0))
    ax.text(-0.72, 1.12, r"60 trading days: $\ln\sigma^{\mathrm{pre}}_{i,t}$",
            ha="center", va="bottom", fontsize=7.5, color="0.3")
    ax.annotate("", xy=(-0.05, 0.4), xytext=(-4.0, 0.4),
                arrowprops=dict(arrowstyle="<->", color="0.4", linewidth=1.0))
    ax.text(-2.0, 0.48, "252 trading days: past return,\nliquidity rule",
            ha="center", va="bottom", fontsize=7.5, color="0.3")
    ax.text(-4.0, -1.2, "Accounting controls: fiscal year $t$ (reported in this 10-K); "
            "text variables: Item 1A of this 10-K.  Not to scale.",
            ha="left", va="center", fontsize=7.5, color="0.3")
    _save(fig, "fig_timeline.pdf")


def main() -> None:
    for f in (fig_volatility_time, fig_coef_fe, fig_horizon_profile, fig_omission_length,
              fig_placebo, fig_timeline):
        f()


if __name__ == "__main__":
    main()

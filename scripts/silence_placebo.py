"""Placebo for H3 (#63): random peer cells, 1,000 permutations.

Within every fiscal year, firms are randomly reassigned to SIC2 cells keeping
the cell sizes of that year; the omission gap is recomputed against these
fake peers (same point-in-time rule and TF-IDF spaces as build_silence.py)
and the H3 model (firm+year FE, h = 30d, common controls) is re-estimated.

Empirical p-value = share of permutations with t >= observed t (one-sided)
and with |t| >= |observed t| (two-sided).
Writes data/silence_placebo.csv and data/results/h3_placebo.csv.
Run:  python scripts/silence_placebo.py [--n 1000] [--jobs 6]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_silence as bs  # noqa: E402
from regressions import CONTROLS, LEN_VAR, DENS_VAR, RESULTS_DIR, load_text_panel, fit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SEED = 20261101
H = "30d"
X = CONTROLS + [LEN_VAR, DENS_VAR, "omission_1a"]


def _shuffled_cells(sec: pd.DataFrame, true_cell_of, rng):
    """cell_of(t) with the year-t cell labels permuted across the year's firms."""
    cache = {}

    def cell_of(t):
        if t not in cache:
            true = pd.Series(true_cell_of(t))
            cache[t] = dict(zip(true.index, rng.permutation(true.to_numpy())))
        return cache[t]
    return cell_of


def one(k, sec, spaces, base):
    rng = np.random.default_rng([SEED, k])
    cell_of = _shuffled_cells(sec, bs.cells_by(sec, "sic_hist"), rng)
    ps = bs.peer_sets(sec, spaces, cell_of)
    om, _, _ = bs.omission_from_sets(sec, spaces, ps, len(sec))
    fake = sec[["ticker", "fiscal_year"]].assign(omission_1a=om)
    d = base.drop(columns=["omission_1a"]).merge(fake, on=["ticker", "fiscal_year"],
                                                 how="inner").dropna(subset=["omission_1a"])
    r = fit(d, X, absorb=("ticker", "fiscal_year"))
    return {"perm": k, "b": float(r.params["omission_1a"]),
            "t": float(r.tstats["omission_1a"]), "p": float(r.pvalues["omission_1a"]),
            "n": int(r.nobs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()

    sec = bs.load_sections("item_1a")
    spaces = bs.prepare_spaces(sec, verbose=False)
    base = load_text_panel(H, xvars=CONTROLS + [LEN_VAR, DENS_VAR])
    real = fit(load_text_panel(H, xvars=X), X, absorb=("ticker", "fiscal_year"))
    t_obs = float(real.tstats["omission_1a"])
    print(f"observed: b={real.params['omission_1a']:+.3f} t={t_obs:+.2f}", flush=True)

    rows = Parallel(n_jobs=a.jobs, verbose=5)(
        delayed(one)(k, sec, spaces, base) for k in range(1, a.n + 1))
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data" / "silence_placebo.csv", index=False)
    summ = pd.DataFrame([{
        "n_perm": len(out), "t_obs": t_obs, "b_obs": float(real.params["omission_1a"]),
        "p_emp_one_sided": float(((out["t"] >= t_obs).sum() + 1) / (len(out) + 1)),
        "p_emp_two_sided": float(((out["t"].abs() >= abs(t_obs)).sum() + 1) / (len(out) + 1)),
        "t_mean": out["t"].mean(), "t_sd": out["t"].std(),
        "t_p95": out["t"].quantile(.95), "t_p975": out["t"].quantile(.975),
        "share_t_gt_1.96": float((out["t"] > 1.96).mean()),
        "b_mean": out["b"].mean(), "seed": SEED}])
    summ.to_csv(RESULTS_DIR / "h3_placebo.csv", index=False)
    print(summ.T)


if __name__ == "__main__":
    main()

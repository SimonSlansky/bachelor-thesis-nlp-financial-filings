"""Panel unit-root tests for short panels (#69, #70).

With T <= 15 annual observations per firm, the relevant risk in a dynamic
panel is Nickell bias (handled in regressions.run_firm_fe), not spurious
regression; the tests below document persistence.

Every variable is first cross-sectionally demeaned (each year's mean
subtracted), which removes common shocks as the year FE do.

* Harris-Tzavalis (1999), fixed T, N -> infinity, balanced panel:
  LSDV estimate of rho in y_it = a_i + rho y_i,t-1 + e_it on firms observed
  in every fiscal year 2012-2024 (T = 13).  H0: rho = 1 for all firms;
  H1: rho < 1 (common).  The null distribution of rho_hat is SIMULATED for
  the actual N and T (random walks, same demeaning); the analytic HT
  normal approximation is reported alongside.
* Im-Pesaran-Shin (2003) t-bar on the unbalanced panel: per-firm ADF
  t-statistics (intercept, p lags) for firms with >= MIN_T observations,
  standardised with E[t] and Var[t] SIMULATED under the null for each
  firm's T.  H0: every firm's series has a unit root; H1: some firms'
  series are stationary.  Rejection does NOT mean every series is I(0).

Writes data/results/stationarity.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from config import DATA_DIR
from regressions import RESULTS_DIR, CONTROLS, TEXT_VARS, _read_panel, _log_pos, _bool
from panel import winsorize_ratios, WINSORIZE_COLS

SEED = 20261101
MIN_T = 8
HT_YEARS = (2012, 2024)
N_SIM = 2000
VARS = ["log_vol", *CONTROLS, *TEXT_VARS, "omission_1a"]


def load() -> pd.DataFrame:
    df = _read_panel()
    df["log_vol"] = _log_pos(df["vol_30d"])
    df["pre_log_vol"] = _log_pos(df["pre_vol"])
    df = df[_bool(df["liquid_pre"])]
    return winsorize_ratios(df, cols=[c for c in WINSORIZE_COLS if c in df.columns])


def matrix(df, var) -> pd.DataFrame:
    m = df.pivot_table(index="fiscal_year", columns="ticker", values=var, aggfunc="mean")
    return m.sub(m.mean(axis=1), axis=0).sort_index()


# ── Harris-Tzavalis ─────────────────────────────────────────────────────

def lsdv_rho(Y: np.ndarray) -> float:
    """Y: T x N balanced.  Within (firm-demeaned) AR(1) coefficient."""
    y, x = Y[1:], Y[:-1]
    y = y - y.mean(axis=0)
    x = x - x.mean(axis=0)
    return float((x * y).sum() / (x * x).sum())


def ht_analytic(T: int):
    """HT (1999) case with individual intercepts; T = number of y_t used."""
    mu = 1 - 3 / (T + 1)
    var = 3 * (17 * T ** 2 - 20 * T + 17) / (5 * (T - 1) * (T + 1) ** 3)
    return mu, var


def harris_tzavalis(mat: pd.DataFrame, rng) -> dict:
    lo, hi = HT_YEARS
    sub = mat.loc[lo:hi].dropna(axis=1, how="any")
    T1, N = sub.shape
    if N < 20:
        return {"ht_N": N}
    rho = lsdv_rho(sub.to_numpy())
    sims = np.empty(N_SIM)
    for s in range(N_SIM):
        rw = np.cumsum(rng.standard_normal((T1, N)), axis=0)
        rw = rw - rw.mean(axis=1, keepdims=True)
        sims[s] = lsdv_rho(rw)
    mu, var = ht_analytic(T1 - 1)
    z = np.sqrt(N) * (rho - mu) / np.sqrt(var)
    return {"ht_N": N, "ht_T": T1, "ht_rho": rho,
            "ht_rho_null_mean_sim": float(sims.mean()),
            "ht_p_sim": float((np.sum(sims <= rho) + 1) / (N_SIM + 1)),
            "ht_z_analytic": float(z), "ht_p_analytic": float(norm.cdf(z))}


# ── Im-Pesaran-Shin ─────────────────────────────────────────────────────

def adf_t(y: np.ndarray, p: int):
    dy = np.diff(y)
    resp = dy[p:]
    cols = [np.ones_like(resp), y[p:-1]] + [dy[p - k:-k] for k in range(1, p + 1)]
    X = np.column_stack(cols)
    dof = len(resp) - X.shape[1]
    if dof <= 1:
        return None
    xtx = np.linalg.pinv(X.T @ X)
    b = xtx @ X.T @ resp
    e = resp - X @ b
    s2 = e @ e / dof
    se = np.sqrt(s2 * xtx[1, 1])
    return float(b[1] / se) if se > 0 else None


_MOM: dict = {}


def null_moments(T: int, p: int, rng):
    if (T, p) not in _MOM:
        ts = [adf_t(np.cumsum(rng.standard_normal(T)), p) for _ in range(N_SIM)]
        ts = np.array([t for t in ts if t is not None])
        _MOM[(T, p)] = (ts.mean(), ts.var())
    return _MOM[(T, p)]


def ips(mat: pd.DataFrame, p: int, rng) -> dict:
    ts, mus, vs, rhos = [], [], [], []
    for c in mat.columns:
        y = mat[c].dropna()
        if len(y) < MIN_T or not (np.diff(y.index.to_numpy()) == 1).all():
            continue
        y = y.to_numpy()
        t = adf_t(y, p)
        if t is None:
            continue
        m, v = null_moments(len(y), p, rng)
        ts.append(t); mus.append(m); vs.append(v)
        x, z = y[:-1] - y[:-1].mean(), y[1:] - y[1:].mean()
        rhos.append((x * z).sum() / (x * x).sum())
    N = len(ts)
    if N == 0:
        return {}
    w = np.sqrt(N) * (np.mean(ts) - np.mean(mus)) / np.sqrt(np.mean(vs))
    return {f"ips{p}_N": N, f"ips{p}_tbar": float(np.mean(ts)),
            f"ips{p}_W": float(w), f"ips{p}_p": float(norm.cdf(w)),
            f"ips{p}_share_firm_reject_5pct": float(np.mean(np.array(ts) < -2.9)),
            "mean_firm_rho": float(np.mean(rhos))}


def main():
    rng = np.random.default_rng(SEED)
    df = load()
    rows = []
    for v in VARS:
        if v not in df.columns:
            print(f"  [skip] {v}")
            continue
        mat = matrix(df, v)
        r = {"var": v}
        r.update(harris_tzavalis(mat, rng))
        r.update(ips(mat, 0, rng))
        r.update(ips(mat, 1, rng))
        rows.append(r)
        print(f"  {v:16s} HT rho={r.get('ht_rho', np.nan):.3f} p_sim={r.get('ht_p_sim', np.nan):.4f} "
              f"| IPS(1) W={r.get('ips1_W', np.nan):+.2f} p={r.get('ips1_p', np.nan):.4f} "
              f"mean rho_i={r.get('mean_firm_rho', np.nan):.2f}")
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "stationarity.csv", index=False)
    print(f"-> data/results/stationarity.csv ({len(out)} variables)")


if __name__ == "__main__":
    main()

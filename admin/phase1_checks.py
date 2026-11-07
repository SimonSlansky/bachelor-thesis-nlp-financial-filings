"""Phase-1 verification checklist (§4.2 of docs/AGENT_GUIDE.md).

Reads the rebuilt data and the pre-Phase-1 backups; prints every number and
writes data/results/phase1_checklist.csv plus the diagnostics CSVs below.
Run after scripts/phase1_run.sh and scripts/check_accessions.py.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from config import DATA_DIR  # noqa: E402

OLD = ROOT / "admin" / "pre_phase1"
RES = DATA_DIR / "results"
RES.mkdir(exist_ok=True)
OUT: list[dict] = []


def rec(item, stat, value, note=""):
    OUT.append({"item": item, "stat": stat, "value": value, "note": note})
    print(f"  [{item}] {stat}: {value}  {note}")


def b(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def main():
    new = pd.read_csv(DATA_DIR / "annual_panel.csv", low_memory=False)
    old = pd.read_csv(OLD / "annual_panel.csv", low_memory=False)
    raw = pd.read_csv(DATA_DIR / "annual_financials.csv", low_memory=False)

    # 1 panel diff
    for nm, d in (("old", old), ("new", new)):
        rec(1, f"{nm} firms", d["ticker"].nunique())
        rec(1, f"{nm} firm-years", len(d))
        rec(1, f"{nm} FY range", f"{d['fiscal_year'].min()}-{d['fiscal_year'].max()}")
    k = ["ticker", "fiscal_year"]
    mg = old[k].merge(new[k], on=k, how="outer", indicator=True)
    rec(1, "firm-years only old / only new / both",
        f"{(mg._merge == 'left_only').sum()} / {(mg._merge == 'right_only').sum()} / "
        f"{(mg._merge == 'both').sum()}")

    # 2 consolidated NI
    m = old[k + ["net_income"]].merge(new[k + ["net_income"]], on=k, suffixes=("_old", "_new"))
    ch = ~np.isclose(m.net_income_old, m.net_income_new, rtol=1e-9, equal_nan=True)
    rec(2, "firm-years with changed net_income (matched)", int(ch.sum()), f"of {len(m)}")
    if {"ni_consol", "ni_parent", "ni_nci"}.issubset(raw.columns):
        r = raw[raw["ni_consol"].notna() & raw["net_income"].notna()]
        rec(2, "net_income == ni_consol where ni_consol exists (raw, before locking)",
            f"{np.isclose(r.net_income, r.ni_consol).mean():.4f}", f"n={len(r)}")
        r = raw.dropna(subset=["ni_consol", "ni_parent", "ni_nci", "total_assets"])
        r = r[r.total_assets > 0]
        ok = ((r.ni_consol - r.ni_parent - r.ni_nci).abs() / r.total_assets) <= 0.01
        rec(2, "share ni_consol - ni_parent ~= ni_nci (within 1% of assets)",
            f"{ok.mean():.4f}", f"n={len(r)}")
        rec(2, "firm-years reporting ProfitLoss (ni_consol)", int(raw["ni_consol"].notna().sum()))
    if "_tag_net_income" in raw.columns:
        rec(2, "net_income tag counts (raw)", json.dumps(raw["_tag_net_income"].value_counts().to_dict()))

    # 3 accession changes (computed by scripts/check_accessions.py)
    acc = DATA_DIR / "accession_changes.csv"
    if acc.exists():
        a = pd.read_csv(acc)
        nt = int(b(a["has_text"]).sum())
        rec(3, "firm-years with stored text in new panel", nt)
        rec(3, "accession changed", int(b(a["changed"]).sum()))
        rec(3, "changed but same-period original 10-K (text valid)", int(b(a["same_period_10k"]).sum()))
        rec(3, "text_mismatch = 1", int(a["text_mismatch"].sum()),
            f"{a['text_mismatch'].sum() / nt:.2%} of text firm-years")

    # 4 shares coverage
    rec(4, "shares_out coverage", f"{new['shares_out'].notna().mean():.4f}")
    rec(4, "shares_out source", json.dumps(new["shares_source"].value_counts().to_dict()))

    vh = pd.read_csv(DATA_DIR / "volatility_horizons.csv", parse_dates=["year_end"])
    p = new[["ticker", "fiscal_year", "year_end", "filing_date", "shares_out"]].copy()
    p["year_end"] = pd.to_datetime(p["year_end"])
    vv = p.merge(vh, on=["ticker", "year_end"], how="left")
    # 5 market cap sanity
    for tk in ("AAPL", "NVDA", "TSLA", "MSFT", "JNJ", "KO"):
        s = vv[vv.ticker == tk].sort_values("fiscal_year").tail(3)
        for _, r in s.iterrows():
            rec(5, f"{tk} FY{r.fiscal_year} filed {r.filing_date}",
                f"mcap=${r.mcap / 1e9:,.0f}bn price_pre={r.price_pre:.2f} shares={r.shares_out / 1e6:,.0f}m")

    # 6 diagnostics
    for f in ("tag_overlap.csv", "tag_locking_summary.csv", "sample_attrition.csv"):
        if (DATA_DIR / f).exists():
            rec(6, f, (DATA_DIR / f).name, pd.read_csv(DATA_DIR / f).to_json(orient="records"))
    for c in [c for c in new.columns if c.startswith("imputed_")]:
        rec(6, f"{c} count", int(b(new[c]).sum()))

    # 7 cut-off
    from volatility import data_cutoff
    rec(7, "data cut-off (last SPY trading day)", str(data_cutoff().date()))
    for fy in (2022, 2023, 2024):
        s = vv[vv.fiscal_year == fy]
        rec(7, f"FY{fy} vol_365d NaN", int(s["vol_365d"].isna().sum()), f"of {len(s)}")

    # 8 volatility horizons
    cov = vh.notna().mean().round(4)
    cov.to_csv(RES / "volatility_horizons_coverage.csv", header=["coverage"])
    rec(8, "coverage of every column", "data/results/volatility_horizons_coverage.csv")
    ob = pd.read_csv(OLD / "volatility_horizons.csv", parse_dates=["year_end"])
    j = ob[["ticker", "year_end", "vol_30d"]].merge(vh[["ticker", "year_end", "vol_30d"]],
                                                    on=["ticker", "year_end"], suffixes=("_o", "_n")).dropna()
    rec(8, "corr(vol_30d new, backup)", f"{j.vol_30d_o.corr(j.vol_30d_n):.5f}", f"n={len(j)}")
    for h in ("30d", "365d"):
        s = vh[[f"vol_{h}", f"idio_vol_{h}"]].dropna()
        rec(8, f"share idio_vol_{h} < vol_{h}", f"{(s.iloc[:, 1] < s.iloc[:, 0]).mean():.4f}")
    vv["fd"] = pd.to_datetime(vv["filing_date"])
    mk = vv.groupby(vv.fd.dt.to_period("M"))["mkt_vol_30d"].median()
    rec(8, "median mkt_vol_30d by filing month, top 3", json.dumps(
        {str(k2): round(v, 3) for k2, v in mk.sort_values(ascending=False).head(3).items()}))
    for h in ("30d", "90d"):
        rec(8, f"mean ea_in_{h}", f"{pd.to_numeric(vh[f'ea_in_{h}'], errors='coerce').mean():.3f}")
    rec(8, "liquid_pre share", f"{b(vh['liquid_pre']).mean():.4f}")

    # 9 text coverage, historical SIC
    tc = DATA_DIR / "text_coverage_by_year.csv"
    if tc.exists():
        t = pd.read_csv(tc, index_col=0)
        rec(9, "Item 1A / Item 7 coverage (all years)",
            f"{t.loc['all', 'item_1a_rate']:.4f} / {t.loc['all', 'item_7_rate']:.4f}")
    pt = pd.read_csv(DATA_DIR / "annual_panel_text.csv", low_memory=False)
    if "sic_at_filing" in pt.columns:
        rec(9, "sic_at_filing coverage", f"{pt['sic_at_filing'].notna().mean():.4f}")
        s = pt.dropna(subset=["sic_at_filing"])
        rec(9, "share SIC4 differs from today", f"{(s.sic_at_filing != s.sic).mean():.4f}")
        rec(9, "share SIC2 differs from today",
            f"{((s.sic_at_filing // 100) != (s.sic // 100)).mean():.4f}")

    # 11 min-years sensitivity (years per firm in the stored panel)
    ny = new.groupby("ticker")["fiscal_year"].nunique()
    rows = [{"min_years": k2, "firms": int((ny >= k2).sum()),
             "firm_years": int(new[new.ticker.isin(ny[ny >= k2].index)].shape[0])}
            for k2 in (3, 5, 7)]
    pd.DataFrame(rows).to_csv(DATA_DIR / "min_years_sensitivity.csv", index=False)
    rec(11, "min_years_sensitivity", json.dumps(rows))

    pd.DataFrame(OUT).to_csv(RES / "phase1_checklist.csv", index=False)
    print("-> data/results/phase1_checklist.csv")


if __name__ == "__main__":
    main()

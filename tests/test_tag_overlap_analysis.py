"""Cross-reference overlap findings with actual tag-switcher firms."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pandas as pd
from config import DATA_DIR

df = pd.read_csv(DATA_DIR / "diagnostics" / "tag_overlap_comparison.csv")
raw = pd.read_csv(DATA_DIR / "annual_financials.csv")

for metric, equiv_tags in [
    ("net_income", {"NetIncomeLoss", "ProfitLoss"}),
    ("operating_cash_flow", {
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    }),
]:
    tag_col = f"_tag_{metric}"
    sub = df[df.metric == metric]
    print(f"\n{'='*70}")
    print(f"  {metric}: {len(sub)} overlap periods across {sub.ticker.nunique()} firms")
    print(f"{'='*70}")

    exact_eq = (sub["diff"] == 0).sum()
    lt1 = (sub["pct_diff"] < 1).sum()
    ge5 = (sub["pct_diff"] >= 5).sum()
    print(f"  Exactly equal:  {exact_eq:4d} ({exact_eq/len(sub)*100:.1f}%)")
    print(f"  Diff < 1%:      {lt1:4d} ({lt1/len(sub)*100:.1f}%)")
    print(f"  Diff >= 5%:     {ge5:4d} ({ge5/len(sub)*100:.1f}%)")

    # Per-firm max diff
    fm = sub.groupby("ticker")["pct_diff"].max()
    print(f"\n  Per-firm max diff:")
    print(f"    Always exactly equal: {(fm == 0).sum()} firms")
    print(f"    Max diff < 1%:        {(fm < 1).sum()} firms")
    print(f"    Max diff < 5%:        {(fm < 5).sum()} firms")
    print(f"    Max diff >= 5%:       {(fm >= 5).sum()} firms")

    # Identify our actual equiv-switchers
    ftags = raw.dropna(subset=[tag_col]).groupby("ticker")[tag_col].nunique()
    actual_sw = set(ftags[ftags > 1].index)
    equiv_sw = set()
    for t in actual_sw:
        used = set(raw.loc[(raw.ticker == t) & raw[tag_col].notna(), tag_col].unique())
        if used.issubset(equiv_tags):
            equiv_sw.add(t)

    overlap_set = set(sub.ticker.unique())
    sw_overlap = equiv_sw & overlap_set
    sw_no_overlap = equiv_sw - overlap_set

    print(f"\n  Our {len(equiv_sw)} tag-switchers:")
    print(f"    Have overlap data: {len(sw_overlap)}")
    print(f"    No overlap (safe, one-tag-per-year): {len(sw_no_overlap)}")

    if sw_overlap:
        sw_sub = sub[sub.ticker.isin(sw_overlap)]
        sw_fm = sw_sub.groupby("ticker")["pct_diff"].max()
        print(f"\n    Among {len(sw_overlap)} switchers with overlap data:")
        print(f"      Max diff == 0%:  {(sw_fm == 0).sum()} firms (identical)")
        print(f"      Max diff < 1%:   {(sw_fm < 1).sum()} firms")
        print(f"      Max diff < 5%:   {(sw_fm < 5).sum()} firms")
        print(f"      Max diff >= 5%:  {(sw_fm >= 5).sum()} firms")

        big = sw_fm[sw_fm >= 5].sort_values(ascending=False)
        if len(big) > 0:
            print(f"\n    ── Detail: switchers with ≥5% difference ──")
            for t in big.index:
                tsub = sw_sub[sw_sub.ticker == t].sort_values("end_date")
                print(f"\n      {t} (max diff {big[t]:.1f}%):")
                for _, r in tsub.iterrows():
                    marker = " ***" if r["pct_diff"] >= 5 else ""
                    print(f"        {r['end_date']}:  tag_A={r['val_a']:>16,.0f}  "
                          f"tag_B={r['val_b']:>16,.0f}  diff={r['pct_diff']:.2f}%{marker}")

        # Also: for these problematic firms, what does the NCI portion look like
        # as a % of total assets?
        if len(big) > 0 and metric == "net_income":
            print(f"\n    ── NCI bias as % of total assets (materiality check) ──")
            for t in big.index:
                tsub2 = sw_sub[sw_sub.ticker == t]
                ta = raw.loc[raw.ticker == t, "total_assets"].median()
                nci = tsub2["diff"].abs().median()
                if ta > 0:
                    nci_pct = nci / ta * 100
                    print(f"      {t}: median |NCI| = {nci:,.0f}, "
                          f"median TA = {ta:,.0f}, "
                          f"NCI/TA = {nci_pct:.2f}%")

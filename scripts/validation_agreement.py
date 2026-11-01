"""Human vs LLM agreement for the extraction audit (#22).

Reads the owner-filled admin/human_labels_template.csv (columns correct_section,
start_ok, end_ok coded 1/0) and data/extraction_validation.csv.  LLM ratings
Pass/Minor = 1, Fail = 0.  Reports per criterion and for "all three OK":
Cohen's kappa, raw agreement, and the human error rate with a Wilson 95% CI.
Writes data/results/validation_agreement.csv.  Exits cleanly if no labels yet.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR  # noqa: E402
from validation_table import wilson_ci  # noqa: E402

LABELS = DATA_DIR.parent / "admin" / "human_labels_template.csv"
OUT = DATA_DIR / "results" / "validation_agreement.csv"
PAIRS = {"correct_section": "correct_section", "start_ok": "start_boundary",
         "end_ok": "end_boundary"}


def cohen_kappa(a, b):
    a, b = pd.Series(a).astype(int).values, pd.Series(b).astype(int).values
    n = len(a)
    if n == 0:
        return float("nan")
    po = (a == b).mean()
    pe = sum(((a == k).mean() * (b == k).mean()) for k in (0, 1))
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def main():
    if not LABELS.exists():
        sys.exit("No label file; run make_human_label_template.py first.")
    h = pd.read_csv(LABELS)
    h = h.dropna(subset=list(PAIRS), how="any")
    if h.empty:
        print("No human labels filled in yet — nothing to compute.")
        return
    llm = pd.read_csv(DATA_DIR / "extraction_validation.csv")
    llm = llm[["ticker", "fiscal_year", "section", *PAIRS.values()]].rename(
        columns={v: f"llm_{v}" for v in PAIRS.values()})
    m = h.merge(llm, on=["ticker", "fiscal_year", "section"], how="inner")
    H = m[list(PAIRS)].astype(int).astype(bool)
    L = m[[f"llm_{v}" for v in PAIRS.values()]].isin(["Pass", "Minor"])
    L.columns = list(PAIRS)
    H["all"], L["all"] = H.all(axis=1), L.all(axis=1)
    rows = []
    for c in H.columns:
        hv, lv = H[c], L[c]
        n = len(m)
        err = int((~hv).sum())
        lo, hi = wilson_ci(err, n)
        rows.append({"criterion": c, "n": n, "agreement": float((hv == lv).mean()),
                     "kappa": cohen_kappa(hv, lv), "human_errors": err,
                     "human_error_rate": err / n, "ci_lo": lo, "ci_hi": hi})
    res = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    res.to_csv(OUT, index=False)
    print(res.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

"""Second, independent LLM audit of the stored extraction sample (#22).

Reviewer: Claude Opus 5.5 (Anthropic), run on 1 Nov 2026 inside the coding
agent, blind to the Gemini verdicts.  Input per section: claimed section, word
count, the first and last 1,200 characters and a 600-character middle excerpt
of the stored text (whitespace collapsed); no surrounding filing.  Same
five-criterion Pass/Minor/Fail rubric as validate_extraction.py.

--collect  merges the reviewer's batch files (local scratch) into
           data/extraction_validation_claude.csv
default    compares the two audits -> data/results/validation_llm_agreement.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR  # noqa: E402
from validation_agreement import cohen_kappa  # noqa: E402
from validation_table import wilson_ci  # noqa: E402

CRITERIA = ["correct_section", "start_boundary", "end_boundary",
            "content_quality", "completeness"]
SRC = DATA_DIR.parent / "admin" / "claude_audit"
OUT = DATA_DIR / "extraction_validation_claude.csv"


def collect():
    idx = pd.read_csv(SRC / "index.csv")
    res = pd.concat([pd.read_csv(p) for p in sorted(SRC.glob("results_batch_*.csv"))])
    for c in CRITERIA:
        res[c] = res[c].astype(str).str.strip().str.capitalize()
        assert res[c].isin(["Pass", "Minor", "Fail"]).all(), c
    d = idx.merge(res, on="audit_id", how="left", validate="one_to_one")
    assert d[CRITERIA].notna().all().all() and len(d) == 200
    d["reviewer"] = "Claude Opus 5.5 (Anthropic)"
    d["input"] = "first/last 1,200 chars + middle 600 chars + word count; no filing context"
    d.to_csv(OUT, index=False)
    print(f"{len(d)} rows -> {OUT}")


def compare():
    g = pd.read_csv(DATA_DIR / "extraction_validation.csv")
    c = pd.read_csv(OUT)
    m = g.merge(c, on=["ticker", "fiscal_year", "section"], suffixes=("_gem", "_cl"))
    rows = []
    for k in CRITERIA + ["all"]:
        if k == "all":
            a = m[[f"{x}_gem" for x in CRITERIA]].isin(["Pass", "Minor"]).all(axis=1)
            b = m[[f"{x}_cl" for x in CRITERIA]].isin(["Pass", "Minor"]).all(axis=1)
        else:
            a = m[f"{k}_gem"].isin(["Pass", "Minor"])
            b = m[f"{k}_cl"].isin(["Pass", "Minor"])
        lo, hi = wilson_ci(int(b.sum()), len(b))
        rows.append({"criterion": k, "n": len(m), "gemini_ok": int(a.sum()),
                     "claude_ok": int(b.sum()), "claude_ok_pct": 100 * b.mean(),
                     "claude_ci_lo": 100 * lo, "claude_ci_hi": 100 * hi,
                     "agreement": float((a == b).mean()), "kappa": cohen_kappa(a, b),
                     "both_fail": int((~a & ~b).sum()), "gemini_only_fail": int((~a & b).sum()),
                     "claude_only_fail": int((a & ~b).sum())})
    out = pd.DataFrame(rows)
    (DATA_DIR / "results").mkdir(exist_ok=True)
    out.to_csv(DATA_DIR / "results" / "validation_llm_agreement.csv", index=False)
    print(out.round(3).to_string(index=False))
    fa = m[~m[[f"{x}_gem" for x in CRITERIA]].isin(["Pass", "Minor"]).all(axis=1)
           | ~m[[f"{x}_cl" for x in CRITERIA]].isin(["Pass", "Minor"]).all(axis=1)]
    pd.set_option("display.width", 220)
    print(fa[["ticker", "fiscal_year", "section", "word_count_gem"] +
             [f"{x}_gem" for x in CRITERIA[:3]] + [f"{x}_cl" for x in CRITERIA[:3]]].to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true")
    a = ap.parse_args()
    if a.collect:
        collect()
    compare()

"""Summarise the stored LLM extraction audit (no API calls, no downloads).

Reads data/validation_sample.csv and data/extraction_validation.csv and writes
  data/results/validation_llm_criteria.csv   per-criterion strict/lenient counts
  data/results/validation_llm_strata.csv     sample design + failures by era x section
  data/results/validation_llm_failures.csv   every section with >= 1 Fail
  tex/tables/ai_validation.tex
With --check-stored it also confirms that the audited texts are the texts in
data/text_sections.csv (same whitespace-token count).
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR  # noqa: E402

ROOT = DATA_DIR.parent
RES = DATA_DIR / "results"
TEX = ROOT / "tex" / "tables" / "ai_validation.tex"

CRITERIA = ["correct_section", "start_boundary", "end_boundary",
            "content_quality", "completeness"]
LABELS = {"correct_section": "Correct section", "start_boundary": "Start boundary",
          "end_boundary": "End boundary", "content_quality": "Content quality",
          "completeness": "Completeness"}
ERAS = [(2010, 2012), (2013, 2015), (2016, 2018), (2019, 2021), (2022, 2024)]


def wilson_ci(k, n, z=1.96):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - s), min(1.0, c + s)


def era_of(fy):
    for lo, hi in ERAS:
        if lo <= fy <= hi:
            return f"{lo}--{hi}"
    return "other"


def check_stored(sample):
    t = pd.read_csv(DATA_DIR / "text_sections.csv")
    m = sample.merge(t, on=["ticker", "fiscal_year"], how="left")
    stored = np.where(m["section"] == "item_1a", m["item_1a"], m["item_7"])
    wc_s = pd.Series(stored).fillna("").str.split().str.len()
    wc_v = m["text"].fillna("").str.split().str.len()
    same = int((wc_s.values == wc_v.values).sum())
    print(f"audited text == stored text (token count): {same}/{len(m)}")
    return same


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check-stored", action="store_true")
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)

    s = pd.read_csv(DATA_DIR / "validation_sample.csv")
    r = pd.read_csv(DATA_DIR / "extraction_validation.csv")
    n = len(r)
    r["strict_all"] = r[CRITERIA].eq("Pass").all(axis=1)
    r["lenient_all"] = r[CRITERIA].isin(["Pass", "Minor"]).all(axis=1)
    r["any_fail"] = r[CRITERIA].eq("Fail").any(axis=1)
    r["era"] = r["fiscal_year"].map(era_of)
    r["empty_extraction"] = r["word_count"].eq(0)

    rows = []
    for c in CRITERIA + ["all"]:
        if c == "all":
            k_s, k_l = int(r["strict_all"].sum()), int(r["lenient_all"].sum())
        else:
            k_s = int(r[c].eq("Pass").sum())
            k_l = int(r[c].isin(["Pass", "Minor"]).sum())
        lo, hi = wilson_ci(k_l, n)
        rows.append({"criterion": c, "n": n, "strict": k_s, "lenient": k_l,
                     "fail": n - k_l, "strict_pct": 100 * k_s / n,
                     "lenient_pct": 100 * k_l / n,
                     "ci_lo": 100 * lo, "ci_hi": 100 * hi})
    crit = pd.DataFrame(rows)
    crit.to_csv(RES / "validation_llm_criteria.csv", index=False)

    s["era"] = s["fiscal_year"].map(era_of)
    design = s.groupby(["era", "section"]).size().unstack(fill_value=0)
    fails = r.groupby(["era", "section"])["any_fail"].sum().unstack(fill_value=0)
    strata = design.add_prefix("n_").join(fails.add_prefix("fail_")).reset_index()
    strata.to_csv(RES / "validation_llm_strata.csv", index=False)

    f = r[r["any_fail"]][["ticker", "fiscal_year", "era", "section", "word_count",
                          "empty_extraction"] + CRITERIA + ["notes"]]
    f.to_csv(RES / "validation_llm_failures.csv", index=False)

    n_filings = s[["ticker", "fiscal_year"]].drop_duplicates().shape[0]
    both = int((s.groupby(["ticker", "fiscal_year"])["section"].nunique() == 2).sum())
    print(f"sections {n}, distinct filings {n_filings}, filings with both sections {both}")
    print(design)
    print(crit.round(1).to_string(index=False))
    print("lenient failures by era:", r.groupby("era")["any_fail"].sum().to_dict())
    print("lenient failures by section:", r.groupby("section")["any_fail"].sum().to_dict())
    print("empty extractions:", int(r["empty_extraction"].sum()))
    if args.check_stored:
        check_stored(s)

    # LaTeX table (second audit columns if available)
    agp = RES / "validation_llm_agreement.csv"
    ag = pd.read_csv(agp).set_index("criterion") if agp.exists() else None
    L = [r"\begin{table}[htbp]", r"  \centering", r"  \small",
         r"  \caption{LLM extraction audits: sections rated acceptable"
         rf" ($n = {n}$ sections, {n // 2} Item~1A and {n // 2} Item~7)}}",
         r"  \label{tab:ai_validation}",
         r"  \begin{tabular}{l r r l r r r}", r"    \toprule",
         r"    & \multicolumn{3}{c}{Gemini 2.5 Flash (full section)} &"
         r" \multicolumn{3}{c}{Claude Opus 5.5 (excerpts)} \\",
         r"    \cmidrule(lr){2-4}\cmidrule(lr){5-7}",
         r"    Criterion & Strict \% & Lenient \% & 95\,\% CI & Lenient \% & Agreement & $\kappa$ \\",
         r"    \midrule"]
    for _, x in crit.iterrows():
        k = x["criterion"]
        name = "Overall (all five)" if k == "all" else LABELS[k]
        if k == "all":
            L.append(r"    \midrule")
        sec = (f"{ag.loc[k, 'claude_ok_pct']:.1f} & {100 * ag.loc[k, 'agreement']:.1f} & "
               f"{ag.loc[k, 'kappa']:.2f}") if ag is not None else "& &"
        L.append(f"    {name} & {x['strict_pct']:.1f} & {x['lenient_pct']:.1f} & "
                 f"[{x['ci_lo']:.1f}, {x['ci_hi']:.1f}] & {sec} \\\\")
    L += [r"    \midrule",
          r"    \multicolumn{7}{l}{\emph{Gemini: sections with at least one Fail, by fiscal-year stratum}} \\"]
    for _, x in strata.iterrows():
        tot = int(x.get("fail_item_1a", 0) + x.get("fail_item_7", 0))
        nn = int(x.get("n_item_1a", 0) + x.get("n_item_7", 0))
        L.append(f"    \\quad FY {x['era']} & \\multicolumn{{6}}{{l}}{{{tot} of {nn}}} \\\\")
    both = "" if ag is None else (
        f" Both audits fail the same {int(ag.loc['all', 'both_fail'])} sections on the overall"
        f" criterion; {int(ag.loc['all', 'gemini_only_fail'])} fail only under Gemini and"
        f" {int(ag.loc['all', 'claude_only_fail'])} only under Claude.")
    L += [r"    \bottomrule", r"  \end{tabular}",
          r"  \begin{minipage}{0.95\linewidth}\vspace{4pt}\footnotesize",
          r"  \emph{Notes.} Strict = rated Pass; lenient = Pass or Minor issue."
          r" The sample is stratified into five three-year fiscal-year strata"
          r" with 20 sections per stratum and section type; Item~1A and Item~7"
          rf" were drawn independently ({n_filings} distinct filings)."
          rf" {int(r['empty_extraction'].sum())} sections with no extracted text"
          r" are counted as Fail on every criterion. Gemini saw the whole extracted"
          r" section; Claude saw its first and last 1{,}200 and a middle 600 characters"
          r" and the word count, blind to the Gemini verdicts. Neither saw the"
          r" surrounding filing. Agreement and Cohen's $\kappa$ on acceptable (1)"
          r" versus Fail (0). Wilson score intervals." + both,
          r"  \end{minipage}", r"\end{table}"]
    TEX.write_text("\n".join(L) + "\n")
    print(f"-> {TEX.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

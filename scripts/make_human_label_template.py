"""Draw the blind human-labelling template for the extraction audit (#22).

Draws a seeded stratified random sample of stored sections from the LLM-audited
sample (data/validation_sample.csv, whose texts are identical to
data/text_sections.csv) plus every non-empty section the LLM failed on
identity or boundaries, so that human-LLM agreement can be computed.
Writes admin/human_labels_template.csv. LLM verdicts are NOT included (blind).
The owner fills correct_section / start_ok / end_ok (1 = yes, 0 = no).
Refuses to overwrite a template that already contains labels.
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR  # noqa: E402

OUT = DATA_DIR.parent / "admin" / "human_labels_template.csv"
DESIGN = DATA_DIR.parent / "admin" / "human_labels_design.csv"  # not for the labeller
SEED = 20261101
PER_CELL = 5          # 5 strata x 2 sections x 5 = 50 sections
ERAS = [(2010, 2012), (2013, 2015), (2016, 2018), (2019, 2021), (2022, 2024)]
CRIT = ["correct_section", "start_boundary", "end_boundary"]


def main():
    if OUT.exists():
        old = pd.read_csv(OUT)
        if old[["correct_section", "start_ok", "end_ok"]].notna().any().any():
            sys.exit(f"{OUT} already contains labels; not overwriting.")
    s = pd.read_csv(DATA_DIR / "validation_sample.csv")
    s = s[s["word_count"] > 0].copy()
    parts = []
    for sec in ("item_1a", "item_7"):
        for lo, hi in ERAS:
            cell = s[(s["section"] == sec) & s["fiscal_year"].between(lo, hi)]
            parts.append(cell.sample(n=min(PER_CELL, len(cell)), random_state=SEED))
    # every section the LLM failed on identity or boundaries (rare cases, so
    # agreement on failures can be assessed at all)
    llm = pd.read_csv(DATA_DIR / "extraction_validation.csv")
    bad = llm[~llm[CRIT].isin(["Pass", "Minor"]).all(axis=1)][["ticker", "fiscal_year", "section"]]
    parts.append(s.merge(bad, on=["ticker", "fiscal_year", "section"]).assign(_flag=1))
    d = (pd.concat(parts).sort_values("_flag", na_position="first")
         .drop_duplicates(["ticker", "fiscal_year", "section"])
         .sample(frac=1, random_state=SEED).reset_index(drop=True))
    txt = d["text"].fillna("")
    out = pd.DataFrame({
        "label_id": range(1, len(d) + 1),
        "ticker": d["ticker"], "fiscal_year": d["fiscal_year"],
        "section": d["section"], "accession_number": d["accession_number"],
        "word_count": d["word_count"],
        "first_400_chars": txt.str[:400].str.replace(r"\s+", " ", regex=True),
        "last_400_chars": txt.str[-400:].str.replace(r"\s+", " ", regex=True),
        "correct_section": "", "start_ok": "", "end_ok": "", "notes": "",
    })
    out.to_csv(OUT, index=False)
    pd.DataFrame({"label_id": out["label_id"],
                  "draw": np.where(d["_flag"].eq(1), "llm_flagged_only", "random")}
                 ).to_csv(DESIGN, index=False)
    print(f"{len(out)} sections -> {OUT}")
    print(out.groupby(["section"]).size().to_dict())


if __name__ == "__main__":
    main()

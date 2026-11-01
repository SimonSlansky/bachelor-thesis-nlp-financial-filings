"""Draw the blind human-labelling template for the extraction audit (#22).

Draws a seeded stratified random sample of stored sections from the LLM-audited
sample (data/validation_sample.csv, whose texts are identical to
data/text_sections.csv) so that human-LLM agreement can be computed.
Writes admin/human_labels_template.csv. LLM verdicts are NOT included (blind).
The owner fills correct_section / start_ok / end_ok (1 = yes, 0 = no).
Refuses to overwrite an existing template so labels are never lost.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import DATA_DIR  # noqa: E402

OUT = DATA_DIR.parent / "admin" / "human_labels_template.csv"
SEED = 20261101
PER_CELL = 5          # 5 strata x 2 sections x 5 = 50 sections
ERAS = [(2010, 2012), (2013, 2015), (2016, 2018), (2019, 2021), (2022, 2024)]


def main():
    if OUT.exists():
        sys.exit(f"{OUT} exists; not overwriting (may contain labels).")
    s = pd.read_csv(DATA_DIR / "validation_sample.csv")
    s = s[s["word_count"] > 0].copy()
    parts = []
    for sec in ("item_1a", "item_7"):
        for lo, hi in ERAS:
            cell = s[(s["section"] == sec) & s["fiscal_year"].between(lo, hi)]
            parts.append(cell.sample(n=min(PER_CELL, len(cell)), random_state=SEED))
    d = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
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
    print(f"{len(out)} sections -> {OUT}")
    print(out.groupby(["section"]).size().to_dict())


if __name__ == "__main__":
    main()

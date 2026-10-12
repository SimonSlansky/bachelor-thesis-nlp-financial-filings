"""Loughran–McDonald sentiment dictionary loader.

Loads the *Loughran–McDonald Master Dictionary* (CSV format) from
``data/lm_master.csv`` and returns Python ``set`` objects for the four
sentiment categories used in this thesis:

* ``positive``     — positive-tone words
* ``negative``     — negative-tone words
* ``uncertainty``  — uncertainty-related words
* ``litigious``    — litigious / legal-risk words

The CSV is the file linked from
https://sraf.nd.edu/loughranmcdonald-master-dictionary/
(file name ``Loughran-McDonald_MasterDictionary_*.csv``).  A column
named, e.g., ``Negative`` contains an integer that is non-zero when the
word belongs to that category (the integer encodes the year in which
the word was added; we treat any non-zero entry as membership).

If ``data/lm_master.csv`` is missing, a clear instruction is printed
explaining how to obtain the file.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

from config import DATA_DIR

LM_PATH = DATA_DIR / "lm_master.csv"

# CSV column names → category keys we use internally.
_CATEGORIES = {
    "positive": "Positive",
    "negative": "Negative",
    "uncertainty": "Uncertainty",
    "litigious": "Litigious",
}


def _instructions() -> str:
    return (
        f"Loughran-McDonald dictionary not found at {LM_PATH}.\n"
        "Download the CSV ('Loughran-McDonald_MasterDictionary_*.csv') from\n"
        "  https://sraf.nd.edu/loughranmcdonald-master-dictionary/\n"
        f"and save it as {LM_PATH}."
    )


def load_lm_dictionary(path: Path = LM_PATH) -> dict[str, set[str]]:
    """Return a dict mapping each category to a set of lowercase words."""
    if not path.exists():
        raise FileNotFoundError(_instructions())

    df = pd.read_csv(path)
    # The master dictionary uses a 'Word' column with uppercase entries.
    word_col = "Word" if "Word" in df.columns else df.columns[0]

    out: dict[str, set[str]] = {}
    for key, col in _CATEGORIES.items():
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' not found in {path.name}; available: "
                f"{list(df.columns)[:10]}…"
            )
        mask = df[col].fillna(0).astype(int) != 0
        words = df.loc[mask, word_col].astype(str).str.lower().str.strip()
        out[key] = set(words[words.str.len() > 1])
    return out


if __name__ == "__main__":  # pragma: no cover
    try:
        lm = load_lm_dictionary()
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    for k, v in lm.items():
        print(f"{k:>12s}: {len(v):>5d} words")

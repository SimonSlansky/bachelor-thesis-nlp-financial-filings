"""Score Item 1A risk-factor sections with FinBERT-tone (CPU).

For every (ticker, fiscal_year) in ``data/text_sections.csv``:

  1. Split Item 1A into sentences with NLTK punkt.
  2. Run ``yiyanghkust/finbert-tone`` (3-class: positive / neutral / negative)
     on each sentence in batches.
  3. Average the per-class probabilities across sentences to obtain a
     document-level score.

Output: ``data/finbert_scores.csv`` with columns
  [ticker, fiscal_year, finbert_neg_1a, finbert_pos_1a,
   finbert_neutral_1a, n_sentences_1a].

Designed to be **resumable** (writes one row at a time, skips already-scored
rows on restart) and **CPU-friendly** (batch inference, torch.no_grad).

Usage:
    .venv\\Scripts\\python.exe scripts/score_finbert.py            # full corpus
    .venv\\Scripts\\python.exe scripts/score_finbert.py --limit 5  # quick test

Runtime estimate on a modern laptop CPU: ~3-5 hours for ~6,000 filings.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA_DIR

MODEL_NAME = "yiyanghkust/finbert-tone"
TEXT_PATH = DATA_DIR / "text_sections.csv"
OUT_PATH = DATA_DIR / "finbert_scores.csv"

BATCH_SIZE = 32           # sentences per forward pass
MAX_TOKENS = 256          # truncation cap per sentence
MIN_SENT_CHARS = 20       # skip ultra-short fragments (timestamps, page nums)
MAX_SENTS_PER_DOC = 400   # cap; longer Item 1A gets uniformly subsampled


def _ensure_punkt() -> None:
    """Make sure NLTK's sentence tokenizer is available."""
    import nltk
    for pkg in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
            return
        except LookupError:
            pass
    nltk.download("punkt_tab", quiet=True)


def _split_sentences(text: str) -> list[str]:
    from nltk.tokenize import sent_tokenize
    if not isinstance(text, str) or not text.strip():
        return []
    sents = sent_tokenize(text)
    out = [s.strip() for s in sents if len(s.strip()) >= MIN_SENT_CHARS]
    if len(out) > MAX_SENTS_PER_DOC:
        # Uniform sub-sample to bound CPU cost on extreme outliers.
        idx = np.linspace(0, len(out) - 1, MAX_SENTS_PER_DOC, dtype=int)
        out = [out[i] for i in idx]
    return out


def _load_already_done() -> set[tuple[str, int]]:
    if not OUT_PATH.exists():
        return set()
    df = pd.read_csv(OUT_PATH, usecols=["ticker", "fiscal_year"])
    return set(zip(df["ticker"].astype(str), df["fiscal_year"].astype(int)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Score only the first N filings (smoke test).")
    args = parser.parse_args()

    if not TEXT_PATH.exists():
        print(f"Missing {TEXT_PATH}; run build_annual_panel.py first.",
              file=sys.stderr)
        sys.exit(1)

    print(f"Loading {TEXT_PATH.name} ...")
    df = pd.read_csv(TEXT_PATH)
    df = df.dropna(subset=["item_1a"]).copy()
    df["fiscal_year"] = df["fiscal_year"].astype(int)
    print(f"  {len(df):,} filings with non-null Item 1A")

    done = _load_already_done()
    if done:
        print(f"  resuming: {len(done):,} filings already scored, skipping")
    todo = df[~df.apply(
        lambda r: (str(r["ticker"]), int(r["fiscal_year"])) in done, axis=1
    )].copy()
    if args.limit is not None:
        todo = todo.head(args.limit)
    print(f"  scoring {len(todo):,} filings")

    if todo.empty:
        print("Nothing to do.")
        return

    print("\nLoading FinBERT-tone (yiyanghkust/finbert-tone) ...")
    _ensure_punkt()
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.eval()

    # FinBERT-tone label order is positive (0) / negative (1) / neutral (2).
    id2label = model.config.id2label
    print(f"  label map: {id2label}")
    label_to_idx = {v.lower(): k for k, v in id2label.items()}
    POS = label_to_idx["positive"]
    NEG = label_to_idx["negative"]
    NEU = label_to_idx["neutral"]

    new_file = not OUT_PATH.exists()
    fout = OUT_PATH.open("a", newline="", encoding="utf-8")
    writer = csv.writer(fout)
    if new_file:
        writer.writerow(["ticker", "fiscal_year",
                         "finbert_neg_1a", "finbert_pos_1a",
                         "finbert_neutral_1a", "n_sentences_1a"])
        fout.flush()

    t0 = time.time()
    n_total = len(todo)
    for i, row in enumerate(todo.itertuples(index=False), 1):
        ticker = str(row.ticker)
        fy = int(row.fiscal_year)
        sents = _split_sentences(row.item_1a)

        if not sents:
            writer.writerow([ticker, fy, "", "", "", 0])
            fout.flush()
            continue

        probs_neg, probs_pos, probs_neu = [], [], []
        with torch.no_grad():
            for j in range(0, len(sents), BATCH_SIZE):
                batch = sents[j:j + BATCH_SIZE]
                enc = tokenizer(batch, padding=True, truncation=True,
                                max_length=MAX_TOKENS, return_tensors="pt")
                logits = model(**enc).logits
                p = torch.softmax(logits, dim=-1).numpy()
                probs_neg.extend(p[:, NEG].tolist())
                probs_pos.extend(p[:, POS].tolist())
                probs_neu.extend(p[:, NEU].tolist())

        writer.writerow([
            ticker, fy,
            f"{np.mean(probs_neg):.6f}",
            f"{np.mean(probs_pos):.6f}",
            f"{np.mean(probs_neu):.6f}",
            len(sents),
        ])
        fout.flush()

        if i % 25 == 0 or i == n_total:
            dt = time.time() - t0
            rate = i / dt if dt > 0 else 0.0
            eta_min = (n_total - i) / rate / 60 if rate > 0 else float("inf")
            print(f"  [{i:>5d}/{n_total}]  {ticker:<6s} FY{fy}  "
                  f"sents={len(sents):>3d}  "
                  f"rate={rate:.2f} filings/s  ETA={eta_min:.1f} min")

    fout.close()
    print(f"\nDone. Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()

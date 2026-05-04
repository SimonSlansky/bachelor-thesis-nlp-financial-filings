#!/usr/bin/env python3
"""
FinBERT scoring of 10-K Item 1A sections — Mac (M-series) edition.

Run on a MacBook with Apple Silicon (M1/M2/M3/M4). Uses the MPS backend
when available, otherwise falls back to CPU.

Model: ``yiyanghkust/finbert-tone`` (Huang, Wang, Yang 2022, CAR).
This variant of FinBERT is pretrained on 4.9 B tokens of corporate filings,
earnings calls, and analyst reports and is the literature standard for
tone analysis of 10-K / 10-Q text.

Inputs  : data/finbert_input_item1a.csv  (cols: ticker, fiscal_year, item_1a)
Outputs : data/finbert_scores_item1a.csv (cols: ticker, fiscal_year,
                                                finbert_neg, finbert_neu,
                                                finbert_pos, finbert_score,
                                                n_chunks)

`finbert_score` = finbert_pos − finbert_neg  (signed tone in [-1, 1])

Setup (one time, in any Python 3.10+ venv on the Mac):
    python -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install torch transformers pandas tqdm

Run:
    python scripts/finbert_score_mac.py

Notes
-----
* Each filing is split into non-overlapping 510-token windows; the per-window
  softmax probabilities are averaged (length-weighted by tokens) to produce
  one (neg, neu, pos) vector per (ticker, fiscal_year).
* Batch size 32 is safe for 16 GB unified memory; raise to 64 if you have
  more headroom.
* The script is resumable: if the output CSV already exists it skips
  (ticker, fiscal_year) rows already scored.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------- config
ROOT          = Path(__file__).resolve().parent.parent
INPUT_CSV     = ROOT / "data" / "finbert_input_item1a.csv"
OUTPUT_CSV    = ROOT / "data" / "finbert_scores_item1a.csv"
MODEL_NAME    = "yiyanghkust/finbert-tone"   # Huang-Wang-Yang 2022 (CAR)
MAX_TOKENS    = 510      # 512 minus [CLS], [SEP]
BATCH_SIZE    = 32
# ----------------------------------------------------------------------


def pick_device() -> str:
    if torch.backends.mps.is_available():
        # allow ops not yet implemented on MPS to fall back to CPU silently
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def chunk_token_ids(ids: list[int], size: int) -> list[list[int]]:
    if not ids:
        return []
    return [ids[i : i + size] for i in range(0, len(ids), size)]


def main() -> None:
    if not INPUT_CSV.exists():
        sys.exit(f"missing input: {INPUT_CSV}")

    device = pick_device()
    print(f"device: {device}")
    print(f"loading {MODEL_NAME} ...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    mdl.to(device)
    mdl.eval()

    # FinBERT label order: 0=positive, 1=negative, 2=neutral  (per model card)
    id2label = {int(k): v.lower() for k, v in mdl.config.id2label.items()}
    pos_idx = next(i for i, v in id2label.items() if v.startswith("pos"))
    neg_idx = next(i for i, v in id2label.items() if v.startswith("neg"))
    neu_idx = next(i for i, v in id2label.items() if v.startswith("neu"))

    df = pd.read_csv(INPUT_CSV)
    df = df.dropna(subset=["item_1a"]).reset_index(drop=True)

    done: set[tuple[str, int]] = set()
    out_rows: list[dict] = []
    if OUTPUT_CSV.exists():
        prev = pd.read_csv(OUTPUT_CSV)
        done = set(zip(prev["ticker"], prev["fiscal_year"]))
        out_rows = prev.to_dict("records")
        print(f"resuming: {len(done):,} rows already scored")

    todo = df[~df.set_index(["ticker", "fiscal_year"]).index.isin(done)]
    print(f"to score: {len(todo):,} filings")

    sep_id = tok.sep_token_id
    cls_id = tok.cls_token_id

    t0 = time.time()
    save_every = 200       # checkpoint cadence (filings)

    for i, row in enumerate(tqdm(todo.itertuples(index=False), total=len(todo))):
        text = row.item_1a
        # tokenize once, then split into 510-token windows
        ids = tok.encode(text, add_special_tokens=False, truncation=False)
        windows = chunk_token_ids(ids, MAX_TOKENS)
        if not windows:
            out_rows.append(
                dict(ticker=row.ticker, fiscal_year=row.fiscal_year,
                     finbert_neg=None, finbert_neu=None, finbert_pos=None,
                     finbert_score=None, n_chunks=0)
            )
            continue

        # build padded batch tensor for this filing
        input_ids: list[list[int]] = []
        attn:      list[list[int]] = []
        for w in windows:
            seq = [cls_id] + w + [sep_id]
            input_ids.append(seq)
            attn.append([1] * len(seq))
        max_len = max(len(s) for s in input_ids)
        pad_id  = tok.pad_token_id or 0
        input_ids = [s + [pad_id] * (max_len - len(s)) for s in input_ids]
        attn      = [a + [0]      * (max_len - len(a)) for a in attn]

        ids_t  = torch.tensor(input_ids, dtype=torch.long, device=device)
        attn_t = torch.tensor(attn,      dtype=torch.long, device=device)

        # forward in mini-batches
        probs_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for s in range(0, ids_t.shape[0], BATCH_SIZE):
                logits = mdl(
                    input_ids=ids_t[s:s + BATCH_SIZE],
                    attention_mask=attn_t[s:s + BATCH_SIZE],
                ).logits
                probs_chunks.append(torch.softmax(logits, dim=-1).cpu())
        probs = torch.cat(probs_chunks, dim=0).numpy()

        # length-weight by actual token count (windows may be unequal at tail)
        weights = [len(w) for w in windows]
        w_sum   = sum(weights)
        avg     = (probs.T @ weights) / w_sum

        out_rows.append(dict(
            ticker        = row.ticker,
            fiscal_year   = int(row.fiscal_year),
            finbert_neg   = float(avg[neg_idx]),
            finbert_neu   = float(avg[neu_idx]),
            finbert_pos   = float(avg[pos_idx]),
            finbert_score = float(avg[pos_idx] - avg[neg_idx]),
            n_chunks      = len(windows),
        ))

        if (i + 1) % save_every == 0:
            pd.DataFrame(out_rows).to_csv(OUTPUT_CSV, index=False)

    pd.DataFrame(out_rows).to_csv(OUTPUT_CSV, index=False)
    elapsed = time.time() - t0
    print(f"done in {elapsed/60:.1f} min  ->  {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

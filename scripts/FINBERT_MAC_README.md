# FinBERT scoring on MacBook (M-series)

This package contains everything needed to score Item 1A sections of 10-K
filings with **FinBERT-tone** (`yiyanghkust/finbert-tone`, Huang-Wang-Yang
2022, *Contemporary Accounting Research*) on a MacBook with Apple Silicon,
then bring the results back to the Windows machine for regression analysis.

This variant of FinBERT is pretrained on 4.9 B tokens of corporate filings,
earnings calls, and analyst reports, making it the accounting-finance
literature's standard for 10-K tone analysis.

## What you need to copy to the Mac

From the project root, copy these two files (preserving the folder layout):

```
data/finbert_input_item1a.csv          (~440 MB, 6,508 filings)
scripts/finbert_score_mac.py
```

You do NOT need the rest of the repo on the Mac.

## One-time setup on the Mac

```bash
cd <wherever you put the project folder>
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch transformers pandas tqdm
```

That's all — `torch` from PyPI on macOS already includes the MPS backend
and FinBERT (~440 MB) downloads automatically on first run.

## Run

```bash
python scripts/finbert_score_mac.py
```

Expected output on first launch:
```
device: mps
loading yiyanghkust/finbert-tone ...
to score: 6,508 filings
100%|████████████| 6508/6508 [25:00<00:00, ...]
done in 25.0 min  ->  data/finbert_scores_item1a.csv
```

The script:
* uses MPS (Apple GPU) automatically; falls back to CPU if MPS missing,
* checkpoints every 200 filings (resumable on Ctrl-C / crash),
* averages per-filing softmax probabilities across all 510-token windows,
  weighted by the number of tokens in each window.

## Output

`data/finbert_scores_item1a.csv` with columns:

| column | meaning |
|---|---|
| ticker, fiscal_year | identifiers |
| finbert_neg | mean P(negative) |
| finbert_neu | mean P(neutral) |
| finbert_pos | mean P(positive) |
| finbert_score | finbert_pos − finbert_neg ∈ [−1, 1] |
| n_chunks | number of 510-token windows averaged |

## Bring back to Windows

Copy `data/finbert_scores_item1a.csv` back into the project's `data/` folder.
The next step on Windows will merge it into the panel and re-run the
regressions to test FinBERT-based H3 candidates.

## Troubleshooting

* `torch.backends.mps.is_available() == False`: macOS < 12.3 or older
  python build. Update macOS or `pip install --upgrade torch`.
* OOM on MPS: lower `BATCH_SIZE` from 32 to 16 or 8 in the script.
* Network blocked when downloading the model: the script needs one-time
  HTTPS access to `huggingface.co`. Run on a network without TLS interception.

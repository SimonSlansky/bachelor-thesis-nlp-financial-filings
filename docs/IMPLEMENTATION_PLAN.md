# Thesis Implementation Plan

**Title:** Textual-Financial Divergence and Stock Return Volatility: Evidence from U.S. Corporate Filings
**Author:** Šimon Slanský
**Submission:** January 2027

---

## 0. Status overview

| Phase | Component | Status |
|---|---|---|
| 1 | Sample selection (top 1,000 → 539 firms after filters) | Done |
| 1 | XBRL annual financial extraction | Done |
| 1 | Per-firm tag locking + equivalence groups | Done |
| 1 | Imputation, ratios, winsorization | Done |
| 1 | Stock returns / annualised volatility (365-day post-filing) | Done |
| 1 | Final panel: 538 firms × 6,322 obs (`data/annual_panel.csv`) | Done |
| 2 | 10-K text extraction (Item 1A, Item 7) | Done |
| 2 | AI quality audit (Gemini 2.5 Flash, 200 sections) | Done |
| 3 | **Textual feature engineering** | **Pending** |
| 3 | **Divergence measure construction** | **Pending** |
| 4 | Baseline regression (financial + lagged vol) | Done |
| 4 | **Text model regression (H1)** | **Pending** |
| 4 | **Divergence model regression (H2)** | **Pending** |
| 5 | **Robustness battery** | **Pending** |
| 6 | Data chapter prose | Done |
| 6 | Results chapter (descriptives + baseline only) | Partial |
| 6 | Methodology / Lit Review / Intro / Conclusion / Robustness | TODO outlines |

---

## 1. Data foundation (DONE)

### 1.1 Sample
- Top 1,000 U.S. firms by SEC market-cap rank
- Filters: operating entities only; SIC 6000–6999 excluded; ≥5 valid XBRL years; ≥90% return coverage
- Result: **539 firms**, FY 2010–2024, **6,639 raw rows / 6,322 regression rows**

### 1.2 XBRL pipeline (`scripts/sec_edgar.py`, `scripts/panel.py`)
- Two-pass amendment-aware extraction with metric-priority resolution
- Per-firm tag locking on multi-tag metrics; equivalence groups for tax-version variants of OCF and net income
- Component imputation (`total_liabilities = total_assets − stockholders_equity`); ±1-period ffill/bfill on balance-sheet items
- Asset growth guarded against multi-year FYE-transition gaps
- Winsorization (1/99) on `leverage, roa, asset_growth, current_ratio, ocf_to_assets`

### 1.3 Returns and volatility (`scripts/returns.py`)
- 365-day calendar window starting 2 calendar days after `filing_date` (`POST_FILING_LAG_DAYS = 2` in `config.py`); on weekends this typically yields a one-trading-day lag, which we accept rather than depending on a market-calendar dependency
- Annualised vol = std of daily log returns × √252
- Coverage filter applied **after** lag-year drop and FY cap to avoid spurious exclusions

### 1.4 10-K text extraction (`scripts/text_parser.py`)
- HTML-DOM parser with bold/uppercase/large-font header detection
- Subtitle-gated, TOC-cluster-aware candidate selection
- Exhibit fallback for "incorporated by reference" stubs
- Output: `data/text_sections.csv` (~6,500 filings × `item_1a`, `item_7`)

### 1.5 AI quality audit (`scripts/validate_extraction.py`)
- 200 stratified samples × 5 criteria (Pass/Minor/Fail) by Gemini 2.5 Flash
- Reported lenient pass: 93.5% overall (Wilson 95% CI: 89.2–96.2%)

---

## 2. Phase 3: Textual feature engineering (NEXT)

All outputs keyed by `(ticker, fiscal_year)` and merged into the panel.

### 2.1 New modules

```
scripts/
├── text_features_dictionary.py    # LM dictionary, TF-IDF, Δtext, risk-factor diff
├── text_features_finbert.py       # FinBERT sentiment scoring (GPU/CPU)
├── build_text_features.py         # orchestrator: produces data/text_features.csv
└── lm_loader.py                   # download & cache LM master dictionary
```

### 2.2 Dictionary-based features (`text_features_dictionary.py`)

Source: Loughran–McDonald master dictionary (latest version), cached to `data/lm_dictionary.csv`.

Tokenisation: lowercase, alphabetic-only, no stemming (LM is unstemmed).

| Feature | Section | Definition |
|---|---|---|
| `lm_neg_pct` | Item 7 | negative-word count / total tokens |
| `lm_pos_pct` | Item 7 | positive-word count / total tokens |
| `lm_net_tone` | Item 7 | (pos − neg) / (pos + neg) |
| `lm_uncertainty_pct` | Item 7 | uncertainty-word count / total tokens |
| `lm_litigious_pct` | Item 1A | litigious-word count / total tokens |
| `lm_risk_density` | Item 1A | (uncertainty + litigious + constraining) / total tokens |
| `text_length_1a` | Item 1A | log token count |
| `text_length_7` | Item 7 | log token count |

### 2.3 TF-IDF and similarity (`text_features_dictionary.py`)

- Per-section TF-IDF vectorizer fit on the **full corpus** (one fit per section, scikit-learn)
- Stop-words: standard English plus a custom list of filing boilerplate
- Min DF = 5 firms, max DF = 95%, unigrams + bigrams

| Feature | Definition |
|---|---|
| `textsim_1a` | cosine similarity of TF-IDF(Item 1A_t) vs Item 1A_{t-1} per firm |
| `textsim_7` | same for Item 7 |
| `delta_text_1a` | 1 − `textsim_1a` |
| `delta_text_7` | 1 − `textsim_7` |

### 2.4 Risk factor diff (`text_features_dictionary.py`)

Item 1A is structured as a list of risk factors separated by sub-headers. Approach:

1. Split Item 1A into risk-factor blocks using heuristic header detection (bold/all-caps short lines).
2. Represent each block as a TF-IDF vector.
3. Match prior-year blocks via greedy maximum bipartite matching above similarity threshold τ = 0.5.
4. Emit:

| Feature | Definition |
|---|---|
| `rf_added` | unmatched current-year blocks |
| `rf_removed` | unmatched prior-year blocks |
| `rf_modified` | matched blocks with similarity < 0.85 |
| `rf_total` | current-year block count |
| `rf_turnover` | (added + removed + modified) / max(rf_total, prior rf_total) |

### 2.5 FinBERT (`text_features_finbert.py`)

Model: `ProsusAI/finbert` (HF transformers).

Process:
1. Sentence-tokenise each section (NLTK punkt).
2. Score each sentence → P(positive), P(negative), P(neutral).
3. Aggregate per (filing, section):
    - `finbert_pos_mean`, `finbert_neg_mean`, `finbert_neu_mean`
    - `finbert_net = pos_mean − neg_mean`
    - `finbert_polarity_share` = share of sentences with max-prob ≠ neutral
4. Truncate sentences > 510 tokens; chunk Item 7 by paragraph if needed.

Compute on Item 7 (primary) and optionally Item 1A. Cache scores per (ticker, FY, section) so re-runs skip done filings.

**Hardware note:** Item 7 averages ~12,000 words ≈ 600 sentences. 6,300 filings × 600 ≈ 3.8M sentence inferences. CPU is feasible (≈6–10 hours with batching) but GPU strongly preferred. Add `--device cuda` flag.

### 2.6 Year-over-year deltas (`build_text_features.py`)

For every dictionary feature `f` and FinBERT score `s`:

```
delta_<feature> = feature_t − feature_{t-1}    (within ticker, sorted by fiscal_year)
```

Resulting wide table: `data/text_features.csv` with columns
`ticker, fiscal_year, <levels>, <deltas>, textsim_*, rf_*`.

### 2.7 Merge into panel

Extend `scripts/build_annual_panel.py` step 11:
- Left-join `text_features.csv` on `(ticker, fiscal_year)`
- Save consolidated `data/annual_panel_full.csv` (regressions read this)

### 2.8 Divergence measure

Add to `scripts/panel.py`:

```python
def add_divergence(df):
    df["delta_roa"] = df.groupby("ticker")["roa"].diff()
    # Continuous: residual from cross-sectional regression of ΔSent on ΔROA
    sub = df.dropna(subset=["delta_lm_net_tone", "delta_roa"])
    lam = OLS(sub["delta_lm_net_tone"], add_constant(sub["delta_roa"])).fit().params["delta_roa"]
    df["divergence"] = df["delta_lm_net_tone"] - lam * df["delta_roa"]
    df["div_dummy"] = (np.sign(df["delta_lm_net_tone"]) != np.sign(df["delta_roa"])).astype(int)
    # Optional FinBERT-based version
    df["divergence_finbert"] = df["delta_finbert_net"] - lam_fb * df["delta_roa"]
    return df
```

Winsorize divergence at 1/99 after construction.

---

## 3. Phase 4: Modelling (extend `scripts/regressions.py`)

All models share: AbsorbingLS with two-digit SIC + fiscal-year FE, firm-clustered SE.

### 3.1 Hypotheses

- **H1:** Δtext predicts vol beyond financials and lagged vol
- **H2:** Divergence predicts vol beyond text + financials

### 3.2 Model ladder

| Model | Specification | Tests |
|---|---|---|
| (1) Baseline-min | σ_{t+1} = α + φ·σ_t + FE | reference |
| (2) Baseline-financial | + size, leverage, ROA, asset_growth | done |
| (3) Text-dictionary | + Δlm_net_tone, Δlm_uncertainty, Δlm_risk_density, delta_text_1a, delta_text_7, rf_turnover | **H1a** |
| (4) Text-FinBERT | model (3) + Δfinbert_net | **H1b** |
| (5) Divergence | model (4) + Divergence | **H2** |

### 3.3 Output tables

Generated to `tex/tables/`:
- `text_dictionary_regression.tex` — model (3)
- `text_finbert_regression.tex` — model (4)
- `divergence_regression.tex` — model (5)
- `model_comparison.tex` — incremental adj-R², F-tests, AIC/BIC across (1)–(5)

### 3.4 Economic magnitude

- 1-SD increase in `divergence` → predicted change in next-year vol
- Quartile-sort portfolio: spread in mean vol between Q1 and Q4 of divergence

---

## 4. Phase 5: Robustness (`scripts/robustness.py`)

| Check | Variation |
|---|---|
| Vol window | 63-day, 180-day in addition to 365-day |
| Alt DV | Post-filing log return |
| Divergence variant | Binary `div_dummy`; OCF-based ΔROA |
| Sentiment source | LM only / FinBERT only / both |
| Subsample | Pre-2020 vs post-2020; tech (SIC 7370–7379, 3674) vs rest |
| Sample filter | Drop firms with <8 years of data |
| Winsorization | 2.5/97.5 instead of 1/99 |
| FE | Add firm FE (within estimator) as alternative to industry FE |

Each writes a row to `tex/tables/robustness_summary.tex` showing the divergence coefficient, t-stat, and adj-R² across specifications.

---

## 5. Phase 6: Writing

Order chosen so each chapter can cite the next:

1. **Methodology chapter** — equations, estimator, FE rationale, fold in volatility-window-overlap and FYE-transition notes (already drafted in TODO comments).
2. **Results chapter** — sub-sections H1, H2, economic magnitude. Reuse generated `.tex` tables.
3. **Robustness chapter** — narrate `robustness_summary.tex`.
4. **Literature review** — write last so each citation is justified by an actually-used method.
5. **Introduction** — finalise after results are known.
6. **Conclusion** — summary, limitations (already noted: dictionary limitations, no causal ID, US-cap bias), future work (FinBERT contextual extension already partly addressed).
7. **Acknowledgments** in `prace.tex`.
8. **Re-enable Czech abstract block** in `zacatek.tex` (required by VŠE template before binding).

---

## 6. File map after implementation

```
scripts/
├── config.py
├── sec_edgar.py
├── returns.py
├── panel.py                      # + add_divergence()
├── text_parser.py
├── validate_extraction.py
├── lm_loader.py                  # NEW
├── text_features_dictionary.py   # NEW
├── text_features_finbert.py      # NEW
├── build_text_features.py        # NEW
├── build_annual_panel.py         # extended with text-merge step
├── regressions.py                # extended with H1/H2 runners
└── robustness.py                 # NEW

data/
├── annual_panel.csv              # financial-only panel (kept for diagnostics)
├── annual_panel_full.csv         # NEW: financial + text features
├── text_sections.csv
├── text_features.csv             # NEW
├── lm_dictionary.csv             # NEW (cached)
└── ... (existing files)

tex/tables/
├── baseline_regression.tex
├── summary_statistics.tex
├── correlation_matrix.tex
├── text_dictionary_regression.tex   # NEW
├── text_finbert_regression.tex      # NEW
├── divergence_regression.tex        # NEW
├── model_comparison.tex             # NEW
└── robustness_summary.tex           # NEW
```

---

## 7. Dependencies to add

```
scikit-learn>=1.3      # TF-IDF, cosine similarity
nltk>=3.8              # sentence tokeniser, stopwords
torch>=2.0             # FinBERT backend
transformers>=4.40     # FinBERT model
sentencepiece          # FinBERT tokeniser
tqdm                   # progress bars for long FinBERT runs
```

---

## 8. Suggested commit sequence (during implementation)

| # | Files | Message |
|---|---|---|
| 1 | requirements.txt | `CHORE: add NLP dependencies (scikit-learn, nltk, transformers)` |
| 2 | scripts/lm_loader.py | `FEAT: add LM dictionary loader with disk cache` |
| 3 | scripts/text_features_dictionary.py | `FEAT: dictionary sentiment, TF-IDF and Δtext features` |
| 4 | scripts/text_features_dictionary.py | `FEAT: risk-factor add/remove/modify diff` |
| 5 | scripts/text_features_finbert.py | `FEAT: FinBERT sentence-level sentiment scoring` |
| 6 | scripts/build_text_features.py | `FEAT: orchestrate text-feature build with YoY deltas` |
| 7 | scripts/panel.py, build_annual_panel.py | `FEAT: divergence measure and merge into panel` |
| 8 | scripts/regressions.py | `FEAT: H1 text-dictionary regression and table` |
| 9 | scripts/regressions.py | `FEAT: H1b FinBERT regression and table` |
| 10 | scripts/regressions.py | `FEAT: H2 divergence regression and table` |
| 12 | scripts/robustness.py | `FEAT: robustness battery (windows, subsamples, alt divergence)` |
| 13 | tex/chapters/04_methodology.tex | `DOCS: write methodology chapter` |
| 14 | tex/chapters/05_results.tex | `DOCS: write H1/H2 results sections` |
| 15 | tex/chapters/06_robustness.tex | `DOCS: write robustness chapter` |
| 16 | tex/chapters/02_literature_review.tex | `DOCS: write literature review` |
| 17 | tex/uvod.tex, tex/zaver.tex | `DOCS: finalise introduction and conclusion` |
| 18 | tex/zacatek.tex, tex/prace.tex | `DOCS: enable Czech abstract and acknowledgments for binding` |

---

## 9. Risk register

| Risk | Mitigation |
|---|---|
| FinBERT runtime on CPU | Cache per-filing scores; allow `--resume`; document GPU instructions |
| Risk-factor splitter fragile on old filings | Fall back to whole-section TF-IDF when block count < 3 |
| Divergence measure colinear with ΔSent | Use residual construction (regress out ΔROA explicitly) |
| Text features missing for first FY per firm | Document in Methodology; regression sample shrinks ~10% |
| Sentence split errors on tables/numbers | Drop sentences <5 alphabetic tokens before FinBERT |
| Reproducibility | Set seeds for TF-IDF tokenizer and any sampling; pin model revisions |

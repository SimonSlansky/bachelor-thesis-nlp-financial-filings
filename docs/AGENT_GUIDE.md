# Agent Guide — Thesis Revision, Phases 0–7

This guide tells an AI coding agent how to do the full examiner-driven revision of the
bachelor thesis *"Item 1A risk-factor disclosure and post-filing volatility"*, safely and
in a form that can be reproduced. Read it completely before touching anything.

- **Repository:** `bachelor-thesis-nlp-financial-filings` (owner SimonSlansky), branch
  `finbert-mac`.
- **Original working copy:**
  `/Users/simon2/Library/CloudStorage/OneDrive-Personal/UČENÍ/Bakalarka`.
- **Other clone:** if you work in a different clone or repo, map every path below to its
  root. `data/` must contain the files listed in §2.3. Copy them; never re-download them.

---

## 0. Hard rules (non-negotiable)

1. **NEVER re-download 10-K filings or re-extract 10-K text.**
   - `data/text_sections.csv` (about 6,600 rows, Item 1A + Item 7) is final.
   - Do not call `text_parser.build_text_dataset`, `text_parser.extract_filing_text`,
     `_download_html`, or `backfill_filing_metadata.py --with-section-source`.
   - Do not write any new code that fetches `sec.gov/Archives/.../*.htm` documents.
   - If a step seems to need fresh text, stop and report. Use the existing text, or record
     the issue as a limitation.
2. **Allowed network calls:**
   - SEC JSON APIs (`data.sec.gov/submissions`, `data.sec.gov/api/xbrl/companyfacts`);
   - SEC filing *index header* pages (`*-index-headers.html`, a few KB, used only for the
     historical SIC);
   - Yahoo Finance via `yfinance`.

   Every SEC request needs `User-Agent: Simon Slansky simon.slansky@outlook.com`
   (`config.SEC_HEADERS`) and must stay below 10 requests/s (`REQUEST_SLEEP=0.15`).
3. **Never refresh existing price files** in `data/prices/`. They are a frozen cache, and
   refreshing changes historical results. Only add files that are missing.
4. **Never delete backups:** `admin/pre_phase1/`, `admin/pre_tier1/`, `admin/pre_tier2/`.
   Before overwriting any `data/*.csv` for the first time in a phase, copy it to
   `admin/pre_phaseN/`.
5. **Python:** use `~/.venvs/bakalarka/bin/python`, a local venv outside OneDrive.
   - The in-repo `.venv` times out on OneDrive reads; do not use it.
   - If you are on another machine, create a venv **outside** any cloud-synced folder and
     run `pip install -r requirements.txt`.
6. **Long jobs** run with `nohup … > admin/<name>.log 2>&1 &`. Poll in cycles of at most
   10 minutes. Before starting a job, check with `ps aux | grep python` that the same job is
   not already running.
7. **Generated outputs:**
   - Every number in the thesis must come from a script output (CSV or `.tex` table).
     Never hand-edit numbers in `tex/tables/*.tex`; regenerate them.
   - Never invent citations. If a reference cannot be verified (DOI, journal, year, pages),
     mark it `% TODO verify` and list it in the handover.
8. **Stop and ask the owner when a step would:**
   - drop more than 5% of the sample;
   - change a hypothesis verdict;
   - need new data sources;
   - conflict with this guide.
9. **Commits:** small and logical, with messages like `phase3: add Holm adjustment to H3`.
   Do not commit `data/prices_raw/` if it exceeds 50 MB; `.gitignore` it instead.
10. **Logging:** append results, decisions and open issues to `docs/revision_log.md` after
    every phase.

---

## 1. Project overview

### 1.1 Research design (current)

- **Sample:** frozen universe of 548 large non-financial US firms (6,757 firm-years, fiscal
  years 2010–2024); 30-day estimation sample 512 firms, 5,782 firm-years.
- **Outcome:** `ln σ_{i,t+1}^{[h]}`, the annualised SD of daily log returns over `h`
  calendar days. The window starts 2 trading days after day 0, the first trading day on or
  after the 10-K filing date. Headline `h = 30`; robustness `h ∈ {5, 10, 90, 180, 365}`.
- **H1 (volume):** `ln(Words^{1A})`, predicted positive.
- **H2 (specificity):** `RiskDensity = (n_unc + n_lit)/Words`, using the LM uncertainty
  and litigious lists. The second stage splits it into `UncDensity` and `LitDensity`.
  - Prediction after Phase 0: **two-sided**.
- **H3 (omission gap):** `Omission = 1 − (peer TF-IDF mass on terms the firm uses)/(total
  peer mass)`. Peers are the leave-one-out SIC2 × year cell with at least 5 peers.
  Predicted positive, estimated under SIC2+year FE and under firm+year FE.
- **Estimator:** `linearmodels.AbsorbingLS`, FE absorbed, two-way clustered SEs by
  (ticker, fiscal_year).

### 1.2 Phase 0 decisions (binding)

| # | Decision |
|---|---|
| 1 | Keep the frozen universe (`data/universe.csv`). Drop the forward-looking post-filing 90% coverage filter. Use the pre-filing liquidity rule `liquid_pre` (≥ 200 closes in the 252 market days before filing). Document the look-ahead / survivorship bias. |
| 2 | **One common control vector for H1–H3:** `ln(pre_vol)`, `log_mcap`, `leverage`, `roa`, `asset_growth`, `bm`, `ret_pre_12m`, `current_ratio`, `ocf_to_assets`. |
| 3 | H2 is two-sided. The literature review lays out both channels. |
| 4 | H3: symmetric decision rule; never say "pre-specified"; Holm adjustment across 5 horizons × 2 FE structures. |
| 5 | Headline `h = 30` days, justified from the literature; 365d reported too. |
| 6 | **No 10-K re-download (owner decision).** #26 (exhibit-fallback share) becomes a documented limitation, not a measured quantity. |

### 1.3 Key early finding (motivates the whole revision)

Adding pre-filing volatility to the 30-day model gave (N = 5,811):

| Variable | Coefficient (t) |
|---|---|
| ln Words 1A | +0.057 (4.62), down from +0.126 |
| LitDensity | −2.51 (−2.24), down from −6.69 |
| Composite RiskDensity | −1.92 (−1.75) |
| Lagged post-filing vol | 0.016 (0.35), i.e. useless |

Expect every effect size and verdict in Chapters 5–6 to change.

---

## 2. Repository map

### 2.1 Scripts (`scripts/`)

| Script | Role | Network? |
|---|---|---|
| `config.py` | All settings: paths, sample rules, XBRL tag map, price settings, liquidity rule | – |
| `sec_edgar.py` | Universe, company metadata, point-in-time XBRL facts, shares by accession | SEC JSON |
| `panel.py` | Tag locking, overlap diagnostics, imputation (with flags), ratios, winsorisation | – |
| `build_annual_panel.py` | Phase-1 driver: XBRL → `annual_panel.csv`, attrition log. Run with `--skip-text` or `--from-raw` | SEC JSON |
| `returns.py`, `volatility.py` | Window machinery, market/idio vol, pre-filing stats, market cap price | Yahoo (cache) |
| `fetch_market_data.py` | SPY plus `data/prices_raw/` (split history) | Yahoo |
| `fetch_earnings_dates.py` | 8-K Item 2.02 dates → `earnings_dates.csv` | SEC JSON |
| `backfill_filing_metadata.py` | Historical SIC from index-header pages → `filing_metadata.csv` | SEC header pages only |
| `build_volatility_horizons.py` | All horizons, lags, market/idio vol, EA dummies, pre-filing vars, mcap, B/M | – (cache) |
| `text_parser.py` | **Frozen. Do not run extraction.** Read only, for documentation | ✗ |
| `text_features.py`, `lm_loader.py`, `build_text_features.py` | LM scoring, counts, TF-IDF YoY similarity, merges → `annual_panel_text.csv` | – |
| `build_silence.py`, `silence_table.py`, `silence_robust_table.py`, `silence_placebo.py` | H3 omission measure and tables | – |
| `regressions.py` | H1/H2 tables (`main_30d`, `main_365d`, `horizons`, `subperiods`, `firm_fe`, `summary`, `correlation`) | – |
| `stationarity.py` | Unit-root section | – |
| `make_figures.py` | PDF figures → `tex/img/` | – |
| `validate_extraction.py` | LLM extraction audit (Gemini) | Gemini API (`keys.txt`) |
| `phase1_run.sh` | Phase-1 orchestrator (text extraction step disabled) | see above |

### 2.2 Thesis (`tex/`)

- **Main file:** `prace.tex`.
- **Chapters:** `uvod.tex` (intro, EMPTY), `chapters/02_literature_review.tex` (EMPTY),
  `03_data`, `04_methodology`, `05_results`, `06_robustness`, `zaver.tex` (conclusion,
  EMPTY).
- **Appendices:** `app01.tex` (variables / diagnostics), `app02.tex` (AI-usage
  documentation, required by the university).
- **Bibliography:** `bibliography.bib` (biblatex).
- **Inputs:** `tables/*.tex` are generated, `figures/` holds TikZ, `img/` holds PDFs.
- **Build:** `cd tex && latexmk` (see `latexmkrc`).

### 2.3 Data (`data/`) — must exist before you start

- **Irreplaceable (copy into any new clone; never regenerate):**
  - `text_sections.csv`
  - `lm_master.csv`
  - `universe.csv`
  - `cik_map.csv`
  - `prices/*.csv`
  - `extraction_validation.csv`
  - `validation_sample.csv`
- **Regenerable:** `annual_financials.csv`, `annual_panel.csv`, `annual_panel_text.csv`,
  `volatility_horizons.csv`, `silence*.csv`, diagnostics.

---

## 3. Issue register (from the examiner review)

The severity labels are the reviewer's. "Phase" is where the issue gets fixed. Text-only
fixes are in Phase 5.

| # | Sev. | Topic | Phase |
|---|---|---|---|
| 1 | Crit | Intro, literature review, conclusion missing | 6 |
| 2 | Min | Informal headings / voice | 5 |
| 3 | Mod | "Every transformation documented" overclaim; price source unnamed | 1 + 5 |
| 4 | Maj | Universe ranking source/date; look-ahead | 1 + 5 |
| 5 | Maj | Post-filing 90% filter selects on the outcome | 1 + 3 |
| 6 | Mod | 5-year minimum justification; sensitivity | 1 + 3 |
| 7 | Mod | "Firm FE" stated for an industry-FE design | 5 |
| 8 | Mod | Only ~15 year clusters; wild bootstrap | 3 |
| 9 | Min | Verify Campbell et al. sample size | 6 |
| 10 | Min | "Triple" → "quadruple" | 5 |
| 11 | Mod | OCF called a "robustness check" | 5 |
| 12 | Mod | 10-Q pilot (15 firms) and S-K citation | 5 |
| 13 | Maj | NetIncomeLoss/ProfitLoss NCI inconsistency | 1 ✅ code |
| 14 | Mod | Tag-difference tails | 1 |
| 15 | Mod | "Materially" undefined | 1 |
| 16 | Mod | Tag-locking losses / look-ahead | 1 |
| 17 | Mod | Imputation counts / sensitivity | 1 + 3 |
| 18 | Maj | Point-in-time XBRL; accession matching | 1 |
| 19 | Mod | 4 vs 5 controls inconsistency | 5 |
| 20 | Mod | Sample-attrition table | 1 + 4 |
| 21 | Maj | Validation stratification arithmetic; failure counts | 2 |
| 22 | Maj | LLM audit validity; human κ; model input | 2 |
| 23 | Mod | Actual coverage instead of "expected 97%" | 1 |
| 24 | Mod | Unbenchmarked "fewer false positives" claim | 2 (drop the claim) |
| 25 | Min | Item 1C date | 5 |
| 26 | Mod | Exhibit fallback share | **Limitation only (no re-download)** |
| 27 | Maj | Cleaning/tokenisation/LM-version documentation | 1 + 5 |
| 28 | Min | 8,800 vs 8,600 words | 4 + 5 |
| 29 | Mod | Timing notation / timeline figure | 4 + 5 |
| 30 | Maj | Fiscal-year FE ≠ calendar shocks; market vol; filing-month FE; idio vol | 1 + 3 |
| 31 | Crit | Pre-filing volatility control | 1 + 3 |
| 32 | Maj | Missing controls (mcap, B/M, past return, EA) | 1 + 3 |
| 33 | Maj | Ratio regression; count specification | 1 + 3 |
| 34 | Crit | H2 sign derivation | 6 + 5 |
| 35 | Mod | "Identification" vs precision wording | 5 |
| 36 | Maj | 365d cut-off inconsistency | 1 + 3 |
| 37 | Maj | Sources of within-firm omission variation | 3 |
| 38 | Maj | Omission ≈ length (r −0.85 vs −0.93) | 3 |
| 39 | Mod | TF-IDF specification | 3 + 5 |
| 40 | Maj | Strategic-silence theory / alternatives | 6 |
| 41 | Maj | H3 control swap; promised footnote check | 3 |
| 42 | Maj | H3 decision rule; multiple testing | 3 + 5 |
| 43 | Maj | Industry FE ≠ firm FE claim | 5 |
| 44 | Maj | Nickell bias | 3 |
| 45 | Mod | Wrong clustering rationale | 5 |
| 46 | Mod | "log is bounded"; RiskDensity max | 4 + 5 |
| 47 | Mod | H1/H2 timing tests | 3 |
| 48 | Mod | Balanced panel; coverage growth | 3 |
| 49 | Mod | Within-R² | 3 |
| 50 | Mod | "Denominator" interpretation | 5 |
| 51 | Mod | Unsupported "firm-specific text" | 5 |
| 52 | Mod | Regulated-industry confound; finer FE | 3 |
| 53 | Mod | H2 "supported" overclaim | 5 |
| 54 | Mod | Effect-size comparison basis | 3 + 5 |
| 55 | Mod | H3 verdict inconsistency | 5 |
| 56 | Mod | Speculative interpretation | 5 |
| 57 | Mod | "Priced" language | 5 |
| 58 | Min | "Predicted sign" for an insignificant result | 5 |
| 59 | Mod | Justify 30 days | 6 + 5 |
| 60 | Mod | COVID split test | 3 |
| 61 | Maj | "Almost unchanged" (−40%) | 4 + 5 |
| 62 | Mod | Within/between variance | 3 |
| 63 | Maj | 5-permutation placebo | 3 |
| 64 | Maj | Length-residual check fails | 3 |
| 65 | Mod | Decile N drop; tails | 3 |
| 66 | Mod | ≥ 20-peer composition | 3 |
| 67 | Mod | Lagged omission purpose | 5 |
| 68 | Min | "Six" vs five checks | 5 |
| 69 | Maj | Fisher-ADF interpretation; small T | 3 |
| 70 | Mod | Boundedness argument | 5 |
| 71 | Maj | Missing literature | 6 |
| 72 | Min | Historical SIC; alternative industry definitions | 1 + 3 |

---

## 4. Phase 1 — Data pipeline (code mostly done; RUN and VERIFY)

### 4.1 Status

All code edits are in place and compile:
- consolidated NI;
- shares by accession;
- tag diagnostics;
- imputation flags;
- attrition log;
- market/idio vol, EA dummies, pre-filing vars, mcap, B/M;
- cut-off rule;
- count variables;
- coverage by year;
- historical SIC.

`phase1_run.sh` runs: XBRL → market data → EA dates → volatility → (text extraction
SKIPPED) → SIC backfill (header pages only) → text features.

A background run may have been started earlier (log `admin/phase1_run.log`). Check `ps`
first and never run two copies.

**Expected runtimes**

| Step | Runtime |
|---|---|
| XBRL | 15–30 min |
| Raw prices | 10–20 min |
| EA dates | 10–25 min |
| Volatility | 10–30 min |
| SIC headers | ~20 min |
| Text features | 5–10 min |

**Total:** about 1.5–2.5 h. Nothing should exceed 1 h. If something does, investigate:
OneDrive I/O is the usual cause, so copy the data to local disk.

### 4.2 Verification checklist (record every number in `revision_log.md`)

1. **Panel diff** against `admin/pre_phase1/annual_panel.csv`: firms, firm-years, FY range.
2. **Consolidated NI:**
   - count of firm-years where `net_income` changed;
   - `net_income == ni_consol` wherever `ni_consol` exists;
   - share where `ni_consol − ni_parent ≈ ni_nci` (within 1% of assets).
3. **Accession changes:** compare `accession_number` per (ticker, FY) old vs new and write
   `data/accession_changes.csv`.
   - **Do not re-extract text.** For changed pairs, check if the old accession is a 10-K for
     the same period. If it is, the stored text is still valid.
   - Otherwise set `text_mismatch = 1` and exclude those firm-years from the text
     regressions. Report the count; if it is above 2%, stop and ask.
4. `shares_out` coverage ≥ 90%.
5. **Market cap:** sanity-check `mcap` for AAPL around 2023-11-03 (about $2.7–2.8T), NVDA
   and TSLA (splits), and two non-splitters. Fix `as_traded_close` if it is off by a split
   factor.
6. **Diagnostics:** `tag_overlap.csv`, `tag_locking_summary.csv`, imputation flag counts,
   `sample_attrition.csv` (universe of 1,000 → final).
7. **Data cut-off** (`volatility.data_cutoff()`); FY2023–24 `vol_365d` NaN counts.
8. **`volatility_horizons.csv`:**
   - coverage of every column;
   - `vol_30d` correlation with the backup ≥ 0.99;
   - `idio_vol < vol` for almost all rows;
   - `mkt_vol_30d` spikes in spring 2020;
   - `ea_in_30d` ≈ 0.2–0.5 and `ea_in_90d` ≈ 1;
   - `liquid_pre` share.
9. `text_coverage_by_year.csv` (#23). `sic_hist` coverage and the share whose SIC differs
   from today's (#72).
10. **`docs/data_dictionary.md`** (#27). For every variable: definition, source, whether
    it is known at filing, winsorisation. Plus:
    - tokenisation: `[a-z]+`, length ≥ 2, lowercase, < 100 tokens = NaN;
    - LM file version and category sizes;
    - the uncertainty ∩ litigious overlap. If non-zero, RiskDensity double-counts those
      words; state it;
    - the exact post-processing steps from `text_parser._clean_section` and `_parse_html`.

      The current thesis text claiming "removes XBRL tags, footnote markers" is
      **inaccurate**; record what really happens.
11. `min_years_sensitivity.csv`: firms and firm-years at ≥ 3, 5, 7 years.
12. **Smoke test** `admin/phase1_smoke.py` (no table output). It runs specs (a)–(e) of the
    original prompt:
    - (a) original model;
    - (b) common control vector;
    - (c) (b) + market vol + EA;
    - (d) (b) with idio-vol as the dependent variable;
    - (e) (b) with historical SIC2.

**Exit criterion:** every item in the Phase-1 checklist (#3, 4, 5, 6, 13, 14, 15, 16, 17, 18,
20, 23, 27, 30, 31, 32, 33, 36, 72) ✅. #26 is marked "limitation".

---

## 5. Phase 2 — Re-validate extraction WITHOUT re-downloading (#21, #22, #24)

The stored text is the object being validated. Use only `text_sections.csv` and existing
validation files.

1. **Inspect** `validate_extraction.py`, `data/validation_sample.csv` and
   `data/extraction_validation.csv`. Establish what was actually done:
   - how the sample was stratified;
   - the real years;
   - which input the LLM saw (extracted section only? with context?);
   - prompt and temperature;
   - per-criterion results.
2. **Fix the arithmetic (#21).**
   - 2010–2024 has three five-year bins, so "20 per five-year period = 100" is impossible.
   - Recompute the real stratification from `validation_sample.csv`.
   - Recount failures: strict 21, lenient 13 of 200. Find out where the thesis's
     "five of the six" came from.
   - If a clean redraw is needed, draw a new stratified sample **from the stored text**
     (e.g. 5 three-year bins × 20 filings × 2 sections). Seed it and save it to
     `data/validation_sample_v2.csv`.
3. **LLM re-audit (optional):** only if `keys.txt` has a working Gemini key.
   - Input = stored section text plus its first and last 500 characters. The full filing
     is not available, and downloading it is forbidden.
   - Record in `app02.tex` that boundary judgements are therefore made **without the
     surrounding filing**. That is a stated limitation.
4. **Human labelling (#22):**
   - Create `admin/human_labels_template.csv` with about 50 random sections (stratified;
     include LLM "Pass" cases).
   - Columns: ticker, FY, section, first 400 chars, last 400 chars, word count, and empty
     columns `correct_section`, `start_ok`, `end_ok`, `notes`.
   - **The owner fills it in. The agent must NOT fill in human labels.**
   - Add `scripts/validation_agreement.py`, which computes Cohen's κ and the human error
     rate with a Wilson CI once the labels exist.
5. **#24:** recommend deleting the "fewer false positives than plain-text extraction" claim
   in Phase 5. No benchmark is run, because it would need downloads.
6. **Citations:**
   - Gilardi et al. (2023): text classification of short texts only; hedge it.
   - Benoit et al. (2016): crowdsourcing; keep it only as an analogy.

**Deliverables:** corrected validation table generator (writes `tex/tables/ai_validation.tex`
instead of the inline table in `app02.tex`), label template, agreement script.

---

## 6. Phase 3 — Econometrics (`regressions.py`, `build_silence.py`, `silence_*.py`, `stationarity.py`)

### 6.0 Common infrastructure (do this first)

In `regressions.py`:
- `CONTROLS = ["pre_log_vol", "log_mcap", "leverage", "roa", "asset_growth", "bm",
  "ret_pre_12m", "current_ratio", "ocf_to_assets"]` (decision 2).
- Keep `lagged_log_vol` only as an optional extra in robustness.
- `WINSORIZE_COLS` += `bm`, `ret_pre_12m` (`log_mcap` and `pre_log_vol` stay unwinsorised).
- `load_text_panel(...)`:
  - filter `liquid_pre == True`;
  - drop `text_mismatch == 1`;
  - use `vol_365d` (not `vol_next_year`) for 365d;
  - choose the industry code via a parameter `ind="sic2"|"sic2_hist"|"sic3"|"ff48"`;
    the default is historical SIC2 (`sic_hist // 100`).
- `_fit(df, xvars, absorb=("ind","fiscal_year"), cluster=("ticker","fiscal_year"))`, which
  also supports `absorb=("ticker","fiscal_year")` and filing-month×year.
- Return the **within-R²** as well as the adjusted R².
- **One results store:** every model writes its coefficients, SEs, N, firms and R² to
  `data/results/*.csv`. The `.tex` tables and the numbers check (Phase 4) read from there.

The H3 scripts (`silence_table.py`, `silence_robust_table.py`, `silence_placebo.py`) must
import the same `CONTROLS` and the same loader (#41).

### 6.1 H1/H2 analyses

| Analysis | Issue | Spec |
|---|---|---|
| Headline table (5 nested columns) at 30d and 365d | 31, 32 | common controls |
| + `ln mkt_vol_h` + `ea_in_h` | 30, 32 | column / robustness table |
| Filing-month × year FE | 30 | robustness |
| Dependent variable `ln idio_vol_h` | 30 | robustness |
| Count spec: `ln Words`, `log_n_unc_1a`, `log_n_lit_1a` | 33 | robustness |
| Wald test unc = lit | – | keep |
| SIC3 / FF48 FE; excluding utilities (SIC 49) and regulated health (80) | 52 | robustness |
| Interaction with post-2019 dummy (2010–19 vs 2020–24) | 60 | replace the split table |
| Timing: lagged text (t−1) and Δtext specifications | 47 | robustness |
| Balanced sub-panel (firms in every FY 2012–2024) | 48 | robustness |
| Min-years 3 / 5 / 7 | 6 | robustness |
| Drop imputed rows | 17 | robustness |
| One-way firm clustering; wild-cluster bootstrap by year (`wildboottest` package, 9,999 draws, Webb weights) | 8 | report p-values |
| Within-R² and incremental R² per block | 49 | table rows |
| Effect sizes on a common basis: 1 pooled SD and 1 within-firm SD for every text variable, in % volatility | 54 | table |
| Within/between variance decomposition of each text variable | 62 | table |

### 6.2 Firm FE / dynamics (#44)

- Firm+year FE **without** the lagged dependent variable; `pre_log_vol` is the dynamic
  control.
- Report the old spec with the lagged DV for comparison and add a paragraph on Nickell
  bias.
- Optional: system GMM via `pydynpd`. Skip it if installation fails and record that.

### 6.3 H3 rebuild (`build_silence.py` + tables)

1. **Document the TF-IDF settings (#39):** tokens, stop-words, unigrams, `min_df=5`,
   `max_df=0.95`, `sublinear_tf`, per-year vs pooled vocabulary. Add a variant with a
   **pooled fixed vocabulary** (#37).
2. **Length-invariant omission (#38, #64):**
   - for each firm-year, draw K = 20 random contiguous blocks of 3,000 tokens from
     Item 1A (seeded) and recompute omission on each;
   - average them → `omission_fixedlen`;
   - firms with fewer than 3,000 tokens use the whole text and get a flag.

   This uses stored text only.
3. **Decompose Δomission (#37):**
   - own-text change (peers held at t−1);
   - peer change (own text held at t−1);
   - composition change.
   Report variance shares.
4. **Placebo (#63):**
   - 1,000 permutations: shuffle firms across SIC2 cells within the year, keeping cell
     sizes;
   - recompute omission and re-estimate with firm FE;
   - report the empirical p-value and save a histogram;
   - parallelise with `joblib` if slow, with a fixed seed.
5. **Decile column (#65):** explain the N drop from 4,871 to 3,876 and add top/bottom
   decile dummies.
6. **≥ 20 peers (#66):** table of the industry composition retained.
7. **#41:** the common control vector already covers the "baseline controls" check
   promised in the footnote; make the footnote consistent with that.
8. **Decision rule (#42):** Holm-adjust the p-values across 5 horizons × 2 FE structures
   and report raw and adjusted. Verdict rule: supported only if significant after Holm
   under the primary design and robust to `omission_fixedlen`. Write the rule into the
   methodology before looking at the new results, and record a timestamp in
   `revision_log.md`.
9. **Reconcile** the r(omission, length) values (−0.85 vs −0.93) and report one, with its
   sample stated.

### 6.4 Stationarity (#69, #70)

- Replace Fisher-ADF with the Harris–Tzavalis (small T, large N) or Im–Pesaran–Shin test
  (implement it if no package exists; statsmodels lacks panel unit-root tests). Use
  cross-sectionally demeaned data.
- State the null correctly. Fix the variable count.
- Reframe the section: with T ≤ 15 the main risk is Nickell bias (§6.2), not spurious
  regression.

### 6.5 Exit criteria

All of these must be in `data/results/`:
- all models listed;
- H3 verdict computed by the pre-written rule;
- smoke comparison old vs new in `revision_log.md`.

**Any verdict change (H1, H2 or H3) is reported to the owner before Phase 5.**

---

## 7. Phase 4 — Tables and figures

1. Every `tex/tables/*.tex` is regenerated from `data/results/`. Move the inline
   `tag_provenance` table (in `app01.tex`) and the `ai_validation` table (in `app02.tex`)
   into generated files.
2. **New tables:**
   - sample attrition (#20);
   - tag overlap (#14);
   - text coverage by year (#23);
   - variance decomposition (#62);
   - effect sizes (#54);
   - robustness summary;
   - H3 placebo;
   - Holm table.
3. **New figures** (`make_figures.py`):
   - timeline (fiscal year end → filing → day 0 → [+2, +h] window, with the pre-filing
     window marked) (#29);
   - placebo histogram;
   - omission vs length, recomputed.
4. **Captions must match the numbers.** The fixed-effects comparison figure must not say
   "stable" if the coefficient changes by more than 20% (#61).
5. **`scripts/check_numbers.py`:**
   - extract every number from `tex/chapters/*.tex`, `uvod.tex` and `zaver.tex`;
   - match each against `data/results/*.csv` and the summary statistics with a tolerance;
   - print the numbers that don't match.
   This catches #28 (8,800 vs 8,600) and #46 (RiskDensity max). The aim is zero unmatched
   numbers in the end.

---

## 8. Phase 5 — Text corrections in Chapters 3–6 and the appendices

Rules:
- edit prose only after Phase 3/4 numbers are final;
- quote numbers only from generated tables;
- keep British spelling as in the existing text;
- mark each edit with `% REV #<n>` (remove the marks in Phase 7).

| # | Required change |
|---|---|
| 2 | Declarative headings ("Text extraction from 10-K filings"); consistent impersonal or "we" voice |
| 3 | Drop "every transformation documented / no vendor adjustments"; name Yahoo Finance and the adjustments |
| 4, 5 | Describe the universe (source, snapshot date, no market-cap values), the liquidity rule and the look-ahead caveat |
| 6 | Re-justify the minimum years; cite the sensitivity |
| 7, 43 | Correct the FE description everywhere |
| 8 | Discuss the number of year clusters; report the wild-bootstrap results |
| 10 | "Roughly quadruple" |
| 11, 19 | OCF is a control; give a consistent control count |
| 12 | Describe the 10-Q pilot honestly; cite Form 10-Q Part II Item 1A / Reg S-K Item 105 |
| 13–18 | Rewrite §3.3 with consolidated NI, the overlap tails, the material count, locking losses, imputation counts and accession matching |
| 21, 22, 24 | Rewrite the validation paragraphs: true stratification, true failure counts, human κ if available, LLM input limitation; delete the comparative claim |
| 23 | Actual coverage |
| 25 | "On or after 15 December 2023" |
| 26 | Add a limitation: the share of sections from the Exhibit 13 fallback is not measured; mostly early-sample MD&A; Item 1A must appear in the 10-K body |
| 27 | Exact cleaning and tokenisation; LM version; overlap handling |
| 29 | Timing notation plus the timeline figure |
| 30–33 | Methodology: new control vector with citations (Kogan et al. 2009; Loughran & McDonald 2011; Campbell et al. 2014); market vol, idio vol, filing-month FE; ratio-regression discussion (Kronmal 1993) |
| 34 | H2 two-sided, with both channels |
| 35 | "Precision", not "identification" |
| 36 | State the cut-off date; 365d treatment consistent |
| 37–42 | H3 methodology rewritten (TF-IDF spec, fixed-length omission, decomposition, decision rule, Holm); never say "pre-specified" |
| 44 | Nickell paragraph |
| 45 | Clustering rationale = persistent firm shocks plus common year shocks |
| 46, 70 | Correct the ranges; remove "log is bounded"; weaken the boundedness argument |
| 47–49, 52, 60, 62 | Report the new robustness results |
| 50, 51, 56, 57, 58 | Remove or hedge speculation and "priced"; fix the predicted-sign wording |
| 53, 55 | Verdicts match the evidence exactly |
| 54, 61 | Effect sizes on a common basis; "falls by X%", not "almost unchanged" |
| 63–68 | H3 robustness section rewritten from the new results |
| 69 | Stationarity section rewritten |

---

## 9. Phase 6 — Introduction, literature review, conclusion, bibliography

1. **Literature review** (about 3,500–5,000 words). Sections:
   1. Textual analysis of 10-K filings: Li 2008; Loughran & McDonald 2011, 2016; Kogan
      et al. 2009.
   2. Risk-factor disclosure: Campbell et al. 2014; Kravet & Muslu 2013; Filzen 2015; Bao
      & Datta 2014; Hope, Hu & Lu 2016.
   3. Boilerplate and stickiness: Brown & Tucker 2011; Dyer, Lang & Stice-Lawrence 2017;
      Cazier et al. 2021; Beatty et al. 2019.
   4. Disclosure theory and withholding: Verrecchia 1983; Dye 1985; Kothari, Shu &
      Wysocki 2009.
   5. Methods: Petersen 2009; Cameron, Gelbach & Miller 2011; Cameron & Miller 2015;
      Nickell 1981; Kronmal 1993; Hoberg & Phillips 2016.
   6. **Hypothesis development:**
      - H1 derived;
      - H2 two-sided, with both channels;
      - H3 grounded in withholding theory, with non-strategic alternatives listed (#40).
2. **Introduction:** motivation, research question, contribution relative to Kogan et al.
   and Campbell et al., data, summary of findings (exactly the final verdicts), structure.
3. **Conclusion:** findings; limitations (universe look-ahead and survivorship,
   non-causal design, LLM audit, #26, SIC2 peers, small T); further research.
4. **`bibliography.bib`:**
   - add every new reference with DOI;
   - verify author, year, title, journal and pages against Crossref
     (`https://api.crossref.org/works?query=...`; allowed, JSON);
   - check #9 against Campbell et al. 2014;
   - anything unverifiable → `% TODO verify` and the handover list.
5. **Do not fabricate quotes, page numbers or sample sizes from papers.**

---

## 10. Phase 7 — Final consistency pass

1. `check_numbers.py` returns zero unmatched numbers.
2. `grep -rn "REV #" tex/` returns nothing (the edit marks are removed).
3. Every `\ref` resolves. `latexmk` succeeds with no undefined references or citations.
4. Every cross-reference promise is fulfilled (e.g. the H3 footnote, #41).
5. `app02.tex` reflects the final AI usage: LLM audit input, human κ, coding assistants.
6. Read the PDF in full: every hypothesis verdict is identical in the intro, results,
   robustness and conclusion.
7. `README.md`:
   - run order: Phase-1 script → `build_silence.py` → `regressions.py` → silence tables
     → `stationarity.py` → `make_figures.py` → `check_numbers.py` → `latexmk`;
   - environment notes;
   - the frozen-text rule.
8. Write a final summary in `revision_log.md` covering every issue #1–#72:
   ✅ fixed / ⚠️ limitation / ❌ not done, each with a reason.

---

## 11. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `TimeoutError: [Errno 60]` on import or pip | OneDrive on-demand files | Use `~/.venvs/bakalarka`; make the folder "Always keep on this device" |
| Merge yields all-NaN columns | `year_end` dtype mismatch (str vs datetime) | `pd.to_datetime` on both sides; assert coverage after every merge |
| Duplicate (ticker, FY) | Fiscal-year-end change | Existing dedup in `filter_transitions_and_duplicates`; assert uniqueness |
| `mcap` off by 2–20× | Split reversal wrong | Check `prices_raw` Close vs actual historical quote; fix `as_traded_close` |
| SEC 403 | Missing User-Agent or rate limit | Use `SEC_HEADERS`; increase sleep |
| yfinance empty / rate-limited | Yahoo throttling | Retry with backoff; never overwrite existing cache |
| `linearmodels` absorbed-variable error | A regressor is collinear with the FE (e.g. time-invariant under firm FE) | Drop it in that spec and note it |
| A step wants new 10-K text | – | **STOP.** Rule 0.1 |

---

## 12. Handover template (end of each phase)

```
## Phase N — completed <date>
Done: …
Key numbers: N, firms, coverage, …
Results changed vs previous: …
Verdict changes: …
Bugs fixed: …
Limitations added: …
Open questions for owner: …
Files changed: …
Commits: <hashes>
```

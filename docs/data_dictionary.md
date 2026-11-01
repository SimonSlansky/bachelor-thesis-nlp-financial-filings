# Data dictionary

Scope: every variable that enters a regression, a table or a figure of the
thesis. Paths are relative to the repository root. "Known at filing" = the
value was public before day 0 (the first trading day on or after the 10-K
filing date). Winsorisation is always at the 1st/99th percentile **of the
estimation sample of each regression** (`panel.winsorize_ratios`, called in
`regressions.load_text_panel`); the stored CSVs are unwinsorised.

Coverage numbers for the rebuilt data are in `data/results/phase1_checklist.csv`
and `data/results/volatility_horizons_coverage.csv`.

## 1. Sources

| Source | Access | Used for |
|---|---|---|
| SEC company list `company_tickers.json` | frozen in `data/universe.csv` (548 operating, non-financial firms; see §6) | universe |
| SEC Submissions API `data.sec.gov/submissions/CIK*.json` | JSON | company metadata (SIC today, FYE month, entity type), 8-K Item 2.02 dates, form/period of text accessions |
| SEC Company Facts API `data.sec.gov/api/xbrl/companyfacts/CIK*.json` | JSON | all accounting variables, shares outstanding |
| SEC filing index headers `…/{accession}-index-headers.html` | a few KB each | SIC code at the time of each filing (`sic_at_filing`) |
| SEC EDGAR 10-K HTML (primary document; Exhibit 13/99 fallback) | **downloaded once (text frozen 12 Oct 2026, commit 280b73c); never re-downloaded** | Item 1A and Item 7 text in `data/text_sections.csv` |
| Yahoo Finance via `yfinance` | frozen per-ticker cache `data/prices/*.csv` (last close 2026-10-30) | adjusted daily closes; SPY market proxy |
| Yahoo Finance via `yfinance` (`auto_adjust=False`, `actions=True`) | `data/prices_raw/*.csv` | split-adjusted close + split events → as-traded price for market cap |
| Loughran–McDonald Master Dictionary 1993–2025 (CSV from sraf.nd.edu) | `data/lm_master.csv` | dictionary word lists |

Prices: `data/prices/{T}.csv` holds Yahoo Finance closes with
`auto_adjust=True`, i.e. adjusted for splits **and** dividends; returns and
volatilities use these. They are a frozen cache: files are only added, never
refreshed (`volatility.load_closes` reads the cache if present).

## 2. Identifiers and timing

| Variable | Definition | Source | Known at filing |
|---|---|---|---|
| `ticker`, `cik` | Firm identifiers | universe / `cik_map.csv` | – |
| `year_end` | Fiscal period end date of the XBRL facts | Company Facts | yes |
| `fiscal_year` | Calendar year of `year_end`; period ends on 1–7 January count to the previous year (52/53-week years). Derived from the period end only (`sec_edgar.fiscal_year_from_end`); the SEC `fiscalYearEnd` field is today's value and drifts for 52/53-week filers, which had shifted the labels of ~20 firms by one year before this revision | `sec_edgar.fiscal_year_from_end` | yes |
| `filing_date` | EDGAR `filed` date of the **first** filing that reports facts for this period end (original 10-K; a 10-K/A only if no original exists), within 180 days of `year_end` | Company Facts | defines day 0 |
| `accession_number` | Accession of that filing (modal accession of the chosen facts) | Company Facts | – |
| `form_type` | `10-K`, or `10-K/A` if no original 10-K facts exist | Company Facts | – |
| day 0 | First trading day of the SPY calendar on or after `filing_date` | `volatility.window_bounds` | – |
| `filing_month` | Calendar month of `filing_date` | | yes |

Point-in-time rule (#18): values, filing date and accession all come from the
single first filing for the period; restated values published later (10-K/A or
next year's comparatives) are never mixed in.

## 3. Accounting variables (XBRL, `data/annual_panel.csv`)

Tag priority (lower wins) from `config.METRICS_WITH_PRIORITY`; only USD facts
from forms 10-K / 10-K/A; flows need a 350–380-day duration.

| Variable | Tags (priority order) | Notes |
|---|---|---|
| `total_assets` | `Assets` | forward-filled from FY t−1 only if t−1 is the immediately preceding year (flag `imputed_ffill_total_assets`) |
| `total_liabilities` | `Liabilities`; else `total_assets − stockholders_equity` (flag `imputed_liab_identity`); else forward fill from t−1 (flag `imputed_ffill_total_liabilities`) | never back-filled from later years |
| `stockholders_equity` | `StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest`, `StockholdersEquity` | per-firm tag locking |
| `equity_parent` | `StockholdersEquity` | used for book-to-market |
| `net_income` | `ProfitLoss` (consolidated, incl. NCI), `NetIncomeLoss`, `IncomeLossFromContinuingOperations` | consolidated basis (#13); `ProfitLoss`/`NetIncomeLoss` are an equivalence group (no locking between them); `IncomeLossFromContinuingOperations` is locked per firm |
| `ni_consol`, `ni_parent`, `ni_nci` | `ProfitLoss`, `NetIncomeLoss`, `NetIncomeLossAttributableToNoncontrollingInterest` | diagnostics only (`data/annual_financials.csv`) |
| `operating_cash_flow` | `NetCashProvidedByUsedInOperatingActivities`, `…ContinuingOperations` | equivalence group |
| `assets_current`, `liabilities_current` | `AssetsCurrent`, `LiabilitiesCurrent` | |
| `shares_out` | `dei:EntityCommonStockSharesOutstanding` (cover page; classes summed), else `us-gaap:CommonStockSharesOutstanding`; matched to the 10-K **by accession number** | source tag in `_tag_shares_out` |

Tag locking: for metrics with non-equivalent alternative tags, each firm keeps
the tag it reports in most years; values under other tags are set to missing
(losses in `data/tag_locking_summary.csv`). This uses the firm's whole tag
history (a mild look-ahead in the *choice of tag*, not in values).
Paired-tag disagreements: `data/tag_overlap.csv` (differences scaled by total
assets; "material" = |difference| > 1% of total assets, `config.TAG_DIFF_MATERIAL`).

| Ratio | Definition | Known at filing | Winsorised |
|---|---|---|---|
| `log_total_assets` | ln(total assets) | yes | no |
| `leverage` | total liabilities / total assets | yes | yes |
| `roa` | net income / total assets | yes | yes |
| `asset_growth` | (TA_t − TA_{t−1}) / TA_{t−1}, only if year ends are 300–400 days apart | yes | yes |
| `current_ratio` | current assets / current liabilities | yes | yes |
| `ocf_to_assets` | operating cash flow / total assets | yes | yes |
| `bm` | book equity (parent, else total) / `mcap` | yes | yes |

Each firm's first fiscal year is dropped (it only serves as the t−1 anchor for
asset growth and lagged volatility; `data/lag_filings.csv`). Fiscal years are
capped at 2024.

## 4. Price-based variables (`data/volatility_horizons.csv`)

h ∈ {5, 10, 30, 90, 180, 365} calendar days. Annualisation √252.

| Variable | Definition | Known at filing | Winsorised |
|---|---|---|---|
| `vol_{h}d` | SD of daily log returns of adjusted closes inside [start, start + h days], start = day 0 + 2 trading days; ≥ 3/5/14/44/89/199 returns; NaN if the nominal window end is after the data cut-off (last SPY close) | **outcome** | no (logged) |
| `lagged_vol_{h}d` | same window after the FY t−1 filing, truncated the day before the current filing; NaN if FY t−1 is absent | yes | no |
| `mkt_vol_{h}d` | SPY volatility over the outcome window | no (contemporaneous control) | no |
| `idio_vol_{h}d` | annualised SD of residuals of r_i = a + b r_SPY + e estimated inside the outcome window | outcome (robustness DV) | no |
| `ea_in_{h}d` | 1 if an 8-K Item 2.02 filing date falls in the outcome window | yes (scheduled) / proxy | no |
| `pre_vol` | annualised SD of the last 60 daily returns ending the last trading day before `filing_date` (≥ 40 returns) | yes | no (logged: `pre_log_vol`) |
| `pre_vol_252` | same over 252 trading days (≥ 168 returns) | yes | no |
| `ret_pre_12m` | log return over the last 252 trading days before filing (≥ 168 closes) | yes | yes |
| `n_pre_days` | firm closes in the 252 market days before filing | yes | – |
| `liquid_pre` | `n_pre_days ≥ 200` (pre-filing liquidity rule, replaces the old post-filing 90% filter, #5) | yes | – |
| `price_pre` | as-traded close on the last trading day before filing = split-adjusted close × product of later split ratios | yes | – |
| `mcap`, `log_mcap` | `price_pre × shares_out`; ln | yes | no |

## 5. Text variables (`data/annual_panel_text.csv`)

### 5.1 Stored text and how it was produced (frozen; documented from `scripts/text_parser.py`, parsing/cleaning code unchanged since the extraction run)

1. HTML parsed with BeautifulSoup (`html.parser`).
2. **Header detection on the DOM** (`p`, `div`, `td`, `h1`–`h6`): element text ≤ 25 words
   (or a bold/large leading `span` of a longer block); bold, ALL-CAPS or font ≥ 12 pt
   (or `<font size>` ≥ 3); not a "(continued)" header; no hyperlink inside; matches an
   "Item X" regex, or (≤ 12 words, within the first 15 characters) a title regex such as
   "Risk Factors" / "Management's Discussion".
3. A sentinel marker is inserted at each header **before** flattening.
4. Removed: `script`, `style`, `ix:header` (the hidden inline-XBRL header block).
   **Tables**: removed if (a) ≥ 4 non-empty cells, ≥ 2 rows and ≥ 30% of non-empty cells are
   bare numbers/currency, or (b) > 40% of non-whitespace characters are digits or `$ % , . ( ) −`;
   empty tables removed; other ("layout") tables are kept with their text. Markers inside
   removed tables are rescued.
5. Inline tags (`span`, `a`, `b`, `i`, `font`, `sup`, …) unwrapped; text extracted with
   newline separators; runs of spaces/tabs collapsed; blank lines collapsed. All remaining
   HTML/inline-XBRL markup disappears in this conversion (tags, not content).
6. **Section choice**: candidates for the item must have a subtitle keyword within 300
   characters ("risk" for 1A; "management", "discussion" or "md&a" for 7); candidates with
   ≥ 3 other item headers within 300 characters (table of contents) and candidates preceded
   within 30 characters by "see", "refer to", "as described in", "included in" are discarded
   when alternatives exist; among the rest the candidate with the longest bounded section is
   chosen. The section ends at the first terminator header (1B, 1C, 2 for Item 1A; 7A, 8, 9,
   9A, 9B for Item 7) more than 100 characters (item headers) or 2,000 characters (title
   headers) after the start; if none, at the next later item in 10-K order. A section needs
   ≥ 200 whitespace-delimited words.
7. **Exhibit fallback**: if no section of ≥ 200 words is found and the stub (< 500 words)
   contains "incorporat", "refer to" or "annual report", the exhibit named in the stub (or
   EX-13 / EX-99.x, at most 3 files) is parsed with the same procedure. Whether a stored
   section came from an exhibit was not recorded (#26, documented limitation).
8. **Post-processing** (`_clean_section`): removes "⟨n⟩ Table of Contents" running headers,
   standalone "Table of Contents" lines, page footers such as "2019 Form 10-K 59" or
   "Annual Report 33", standalone 1–3-digit page numbers; replaces non-breaking spaces;
   drops lines with ≥ 4 tokens of which ≥ 3 and > 40% are bare numeric tokens (residual
   table rows); in sections of > 50 lines drops short lines (≤ 8 words) that occur ≥ 4 times
   (running headers/footers); collapses blank lines.

Not done (the pre-revision thesis text was inaccurate here): footnote markers are **not**
removed, layout-table text is **kept**, and there is no separate "XBRL tag removal" step
beyond steps 4–5.

Text for 10-K/A rows was not extracted. The stored text is keyed by the
fiscal-year labels of the extraction run; `data/text_accessions.csv` (accessions,
period ends and filing dates of the text filings, from the panel of commit 280b73c,
the last commit that changed `text_sections.csv`) re-keys every text row to the
current fiscal-year rule (`text_features.rekey_text`); the text itself is not
modified. Firm-years whose accession changed in the
point-in-time rebuild are checked in `data/accession_changes.csv`; those whose stored text
is not from an original 10-K for the same period carry `text_mismatch = 1` and are excluded
from text regressions.

### 5.2 Tokenisation and dictionary counts (`scripts/text_features.py`)

- Tokens: lowercase text, regex `[a-z]+`, keep tokens of length ≥ 2. Digits, hyphens and
  apostrophes split tokens.
- A section with < 100 tokens gets NaN for all rates (treated as an extraction failure).
- LM dictionary: Loughran–McDonald Master Dictionary **1993–2025** (`data/lm_master.csv`,
  86,553 entries). Category membership = positive year-added code; negative codes (words
  **removed** in 2020: 10 negative, 7 positive, 2 litigious — "para", "terminus") are
  **not** counted (fixed in this revision; before, they were counted). Category sizes
  used: positive 347, negative 2,345, uncertainty 297, litigious 903.
- Uncertainty ∩ litigious = **∅** (0 words), so RiskDensity does not double-count any token.

| Variable | Definition | Known at filing | Winsorised |
|---|---|---|---|
| `words_1a`, `words_mda` | number of tokens in Item 1A / Item 7 | yes (in the filing) | no |
| `log_words_1a` | ln(max(words_1a, 1)) | yes | no |
| `n_unc_1a`, `n_lit_1a` | LM uncertainty / litigious token counts in Item 1A | yes | no |
| `unc_1a`, `lit_1a` | `n_unc_1a / words_1a`, `n_lit_1a / words_1a` | yes | no |
| `risk_1a` (RiskDensity) | `(n_unc_1a + n_lit_1a) / words_1a` | yes | no |
| `log_n_unc_1a`, `log_n_lit_1a` | ln(1 + count) (count specification, #33) | yes | no |
| `neg_1a`, `sentiment_mda` | LM negative share of Item 1A; (pos − neg)/words of Item 7 | yes | no (auxiliary) |
| `textsim` | cosine similarity of Item 1A TF-IDF vectors (corpus-wide fit, `[a-z]{2,}`, min_df 5, max_df 0.95, sublinear tf) with the firm's FY t−1 Item 1A | yes | no (auxiliary) |

### 5.3 Omission gap (H3, `scripts/build_silence.py`, `data/silence.csv`)

See the module docstring for the exact rule. TF-IDF: scikit-learn `TfidfVectorizer`,
lowercase, token pattern `(?u)\b\w\w+\b` (≥ 2 letters/digits/underscores), scikit-learn
English stop-word list removed, unigrams, `min_df=5`, `max_df=0.95`, raw counts
(`sublinear_tf=False`), smoothed IDF, l2-normalised rows; vocabulary and IDF re-fitted for
each fiscal year on Item 1A documents filed before that year's first filing. Peers: other
panel firms in the same SIC2 cell (historical SIC2 by default), most recent Item 1A of FY t
or t−1 filed strictly before the focal filing; ≥ 5 peers. Known at filing: yes. Not
winsorised (bounded in [0, 1]).

| Variable | Definition |
|---|---|
| `omission_1a` | 1 − peer TF-IDF mass on terms the focal firm uses / total peer mass (historical SIC2 peers) |
| `omission_1a_sicnow` | same with today's SIC2 (pre-revision definition) |
| `omission_fixedlen_1a` | mean over 20 random contiguous 3,000-raw-token blocks of the focal Item 1A (seeded); whole text if shorter (`fixedlen_short_1a = 1`) |
| `omission_pooled_1a` | one vocabulary/IDF fitted on all years (uses later documents; diagnostic only) |
| `commission_1a` | focal TF-IDF mass on terms no peer uses |
| `peer_n_1a` | number of peers |

## 6. Sample rules

- Universe: `data/universe.csv` — 548 tickers frozen from the SEC company list. The file
  holds the firms that passed the operating-company and non-financial (SIC 6000–6999
  excluded) filters of the original run from the top-1,000 list; the original snapshot date
  of the SEC list is not recorded (`universe_meta.json` shows the freeze date of the file).
  The ranking is by today's (snapshot-date) size, not point-in-time → look-ahead /
  survivorship bias (Phase-0 decision 1).
- ≥ 5 fiscal years of 10-K XBRL data at extraction (`config.MIN_VALID_YEARS`).
- Estimation sample: `liquid_pre == True`, `text_mismatch != 1`, all model variables
  non-missing, outcome > 0.

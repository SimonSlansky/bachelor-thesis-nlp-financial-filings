# Revision log (examiner-review fixes)

## Phase 0 decisions (1 Nov 2026)
1. Universe: keep current universe; drop forward-looking 90% post-filing return
   filter in favour of a pre-filing liquidity rule; acknowledge look-ahead bias.
2. One common control vector for H1–H3: log market cap, leverage, ROA, asset
   growth, book-to-market, past 12m return, pre-filing volatility, current
   ratio, OCF/assets.
3. H2 stated two-sided; both theoretical channels laid out.
4. H3: symmetric decision rule, no "pre-specified" wording, Holm adjustment
   across 5 horizons x 2 FE structures.
5. Keep h = 30 days as headline, justified from literature; report 365d.

## Phase 1.3 — pre-filing volatility (#31) — diagnostic result
`pre_vol` = annualised SD of 60 trading-day returns ending the day before filing
(volatility.py::pre_filing_vol). Rebuilt `data/volatility_horizons.csv`
(coverage 99.0%). Merged in memory only (build_text_features.py needs sklearn).

30-day, SIC2+year FE, two-way clustered, N = 5,811 / 508 firms:

| Spec | lagged ln σ | pre ln σ | ln Words1A | UncDens | LitDens | RiskDens |
|---|---|---|---|---|---|---|
| original | 0.137 (1.80) | – | 0.126 (6.84) | −2.36 (−1.11) | −6.69 (−3.91) | |
| + pre_vol | 0.016 (0.35) | 0.564 (10.79) | 0.057 (4.62) | −1.24 (−0.73) | −2.51 (−2.24) | |
| + pre_vol, composite | 0.016 (0.35) | 0.565 (10.74) | 0.055 (4.67) | | | −1.92 (−1.75) |

Implications:
- H1 survives but magnitude halves (0.126 → 0.057).
- LitDensity survives at 5% but shrinks ~60%; composite RiskDensity only 10%.
- Lagged post-filing vol becomes irrelevant → replace it with pre_vol as the
  dynamic control (also mitigates Nickell bias, #44).
- All effect-size statements in Ch. 5–6 must be rewritten after the rebuild.

## H3 decision rule (#42) — written 2026-11-01 12:38:34 CET, before any H3 re-estimation
Written before any H3 model is re-run on the rebuilt data; the verdict is
computed mechanically by `scripts/silence_table.py` (`h3_verdict.json`).

- Specification: eq. (silence_model) with the common control vector, ln Words1A
  and RiskDensity; DV ln σ^[h]; two-way (firm, year) clustered SEs; industry =
  historical SIC2; peers = point-in-time SIC2 cell, ≥ 5 peers.
- Test family for Holm: h ∈ {5, 10, 30, 90, 180} days × {SIC2+year FE,
  firm+year FE} = 10 two-sided tests of β_om = 0; Holm step-down at α = 0.05.
  h = 365 is reported outside the family.
- Primary design: h = 30 days, under both FE structures (treated symmetrically).
- Evidence in FE structure s:
  - positive: β_om(30d, s) > 0 with Holm-adjusted p < 0.05 AND the
    length-invariant `omission_fixedlen_1a` coefficient (same spec, 30d, s) > 0
    with p < 0.05;
  - negative: the mirror image (both < 0, same significance conditions).
- Verdict:
  - **supported** — positive evidence under both FE structures;
  - **partially supported** — positive evidence under exactly one structure and
    no negative evidence under the other;
  - **contradicted** — negative evidence under ≥ 1 structure and no positive
    evidence under either;
  - **not supported** — every other case.
- The 1,000-draw placebo p-value, the decomposition and the other robustness
  columns are reported and discussed, but do not change the label.
- Wording: never "pre-specified"; the rule is described as the decision rule
  stated in the methodology.

## Phase 1 — working notes (1 Nov 2026)
- First background run (12:22) died silently at step 1 (no traceback, log held
  only the header); restarted 12:25 with `caffeinate` + unbuffered output.
- **Bug: fiscal-year labels (new).** `fiscal_year` was derived from today's SEC
  `fiscalYearEnd` field, which drifts for 52/53-week filers (e.g. "0101").
  ~20 firms (DAR, SNA, SWK, TXT, MASI, ILMN, TRMB, LHX, DPZ, EXEL, LSCC, JNJ, TDY,
  USFD, …; 249 raw rows) carried labels one year too high → wrong year FE, wrong
  t−1 matching and lost FY2024 under the FY ≤ 2024 cap; the text-era panel had
  156 such rows. Fix: `fiscal_year = year(year_end − 7 days)`
  (`sec_edgar.fiscal_year_from_end`), and stored text re-keyed by its own
  period end via `data/text_accessions.csv` (text untouched; built from the
  panel of commit 280b73c, the last commit that changed `text_sections.csv`).
- **Bug: removed LM words counted.** `lm_loader` counted words whose LM code is
  negative (= removed in 2020) as members: 2 litigious ("para", "terminus"),
  7 positive, 10 negative. Fixed (`> 0`).
- **Shares coverage.** Multi-class filers (CMCSA, V, MA, META, F, UPS, …) report
  cover-page shares per class only (dimensional facts are absent from Company
  Facts) → shares/mcap coverage 91% of the regression-eligible sample and B/M
  90% → the Phase-0 control vector would drop ~11% (> 5% rule). Fix attempt:
  fallback to the annual `WeightedAverageNumberOfSharesOutstandingBasic` of the
  same accession (flag `shares_source`); book equity falls back to TA − TL.
- Accession check (#18): of 6,587 text firm-years, 7 changed accession (the
  text came from a 10-K/A; the point-in-time rule now picks the original 10-K)
  → `text_mismatch = 1` for 7 (0.11%).
- The existing `data/silence.csv` (thesis numbers) came from an older,
  non-point-in-time version of `build_silence.py` (5,367 valid vs 5,020); the
  refactored `build_silence.py` reproduces the committed point-in-time code to
  1.8e-7.
- 6 universe firms never had Yahoo price files (BLD, CTRA, GTLS, HOLX, MASI,
  PSTG: "possibly delisted"); they had no outcome before either.
- `ea_in_30d` mean = 0.01 (guide expected 0.2–0.5): for most firms the Q4
  earnings release precedes the 10-K and the next one falls > 30 days later, so
  the 30-day window rarely contains an EA; `ea_in_90d` = 0.99. The dummy has
  almost no variation at 30d. Verified: last EA before the 10-K median 11 days
  earlier, next EA median 68 days later (2.3% within 35 days).
- Shares fallback result: shares coverage 97.8%; `log_mcap` 96.7%, `bm` 95.9%
  (were 90.3% / 88.6%). Smoke spec (a) → (b) sample 6,045 → 5,782 (−4.4%, < 5%).
- SIC back-fill: 1,316 index-headers pages do not exist for older filings; the
  filing index page (`-index.html`, ~15 KB) carries the same historical SIC
  (checked on 3 filings where SIC changed) → coverage 99.99%.
- Universe provenance: `data/universe.csv` (548) = survivors among the first
  1,000 rows of `data/cik_map.csv`, the SEC `company_tickers.json` list as
  committed on 2026-10-12 (280b73c); all 548 are within its first 1,000 rows
  (the 2026-04-06 version, c0e9369, contains only 546 of them in its top
  1,000). `scripts/universe_attrition.py` reconstructs the 1,000 → 548 stages.

## Phase 1 — completed 1 Nov 2026 (checkpoint: awaiting owner)
Done: full rebuild (XBRL point-in-time, consolidated NI, shares by accession,
market data, EA dates, volatility horizons + market/idio vol + pre-filing vars
+ mcap/BM, historical SIC, text features); 3 bugs fixed (FY labels, removed LM
words, multi-class shares); accession check; checklist
(`admin/phase1_checks.py` → `data/results/phase1_checklist.csv`); smoke test
(`admin/phase1_smoke.py` → `admin/phase1_smoke.csv`); `docs/data_dictionary.md`.

Checklist (§4.2):
1. Panel: old (pre_phase1, with the 90% filter) 529 firms / 6,483 firm-years;
   new 548 / 6,757; FY 2010–2024 both. Only-old 5, only-new 280, both 6,478
   (matched on ticker, FY).
2. Consolidated NI: net_income changed in 2,547 of 6,483 firm-years (matched by
   period end; median |Δ| = 0.084% of assets; > 1% of assets in 237);
   net_income = ni_consol wherever ProfitLoss exists (4,598/4,598);
   ni_consol − ni_parent ≈ ni_nci within 1% of assets in 99.7% (n = 2,989).
3. Accessions: vs the text-era panel, 7 of 6,587 text firm-years changed
   accession (text from a 10-K/A) → text_mismatch = 1 (0.11%); vs the
   pre_phase1 panel 0 changed (matched by period end); vs the text-era panel
   overall (incl. 10-K/A rows without text) 49. `data/accession_changes.csv`.
4. shares_out coverage 97.8% (≥ 90% ✅).
5. mcap: AAPL filed 2023-11-03 → $2,762bn ✅; NVDA filed 2024-02-21 → as-traded
   $694.52 × 2,500m = $1,736bn (pre 10:1 split) ✅; TSLA filed 2023-01-31 →
   $166.66 × 3,164m = $527bn ✅; MSFT, JNJ, KO plausible. No split errors found.
6. tag_overlap: NI pair (ProfitLoss vs NetIncomeLoss) n = 3,838, median 0.043pp,
   p95 1.48pp, p99 3.56pp, max 37.4pp of assets, material (> 1% of assets) 303
   (7.9%) — this is the NCI share, now resolved by the consolidated basis;
   OCF pair n = 417, median 0.035pp, p99 4.19pp, material 50 (12.0%).
   Locking losses: equity 637 of 7,900 values nulled (171 firms with > 1 tag),
   NI 38 of 7,813, OCF 0. Imputations: liabilities from A − E 2,171; ffill
   assets 1, liabilities 32. Attrition 548 → 6,757 in `sample_attrition.csv`;
   1,000 → 548 in `universe_stages.csv`: 1,000 → 753 operating → 600
   non-financial → 599 with FYE → 550 with ≥ 5 XBRL years (today's metadata);
   all 548 universe firms reproduced, plus ROIV and CRDO, which now have
   exactly 5 years.
7. Cut-off 2026-10-30; FY2022/23/24 vol_365d NaN 9 each (no-price firms only).
8. vol_30d corr with backup 1.0000 (n = 6,344); idio < total in 88.3% (30d) /
   98.9% (365d) — at 30d the residual SD uses n − 2 dof, so for firms with a
   low market R² idio can exceed total; mkt_vol_30d peaks for Feb-2020 filings
   (0.90) and Mar-2020 (0.57); ea_in_30d 0.014, ea_in_90d 0.987; liquid_pre
   98.7%.
9. Text coverage: Item 1A 96.3%, Item 7 94.9% (by year in
   `text_coverage_by_year.csv`); sic_at_filing 99.99%, SIC4 ≠ today 2.2%,
   SIC2 ≠ today 1.6%.
10. `docs/data_dictionary.md` written (tokenisation, LM 1993–2025 file,
    category sizes 347/2,345/297/903, unc ∩ lit = ∅, exact cleaning steps;
    the old "removes XBRL tags, footnote markers" claim is wrong).
11. min_years_sensitivity: ≥3: 544 firms / 6,749; ≥5: 516 / 6,649; ≥7: 487 /
    6,492 (years per firm in the panel; the ≥ 5 XBRL-year rule at extraction
    cannot be relaxed without new 10-K text).
12. Smoke test (30d, decomposition col 5; coef (t)):

| | (a) original | (b) common | (c) +mkt vol, EA | (d) idio DV | (e) hist SIC2 |
|---|---|---|---|---|---|
| ln Words1A | +0.123 (7.44) | +0.058 (4.73) | +0.068 (6.12) | +0.090 (6.27) | +0.058 (4.54) |
| UncDensity | −1.80 (−0.93) | −0.52 (−0.30) | +0.11 (0.08) | −2.63 (−1.58) | −0.55 (−0.31) |
| LitDensity | −6.56 (−3.86) | −2.95 (−2.76) | −3.26 (−3.79) | −2.97 (−2.29) | −3.02 (−2.90) |
| RiskDensity (col 4) | −4.32 (−3.25) | −1.80 (−1.64) | −1.66 (−2.01) | −2.80 (−2.46) | −1.85 (−1.70) |
| ln σ_pre | – | +0.584 (9.90) | +0.554 (13.37) | +0.565 (12.67) | +0.584 (9.85) |
| lagged ln σ | +0.153 (2.18) | – | – | – | – |
| N / firms | 6,045 / 526 | 5,782 / 512 | 5,782 / 512 | 5,782 / 512 | 5,782 / 512 |

Key numbers: N(30d, common controls) = 5,782 firm-years / 512 firms.
Results changed vs previous: H1 halves (0.123 → 0.058) but stays significant;
composite RiskDensity loses 5% significance (t −3.25 → −1.64); LitDensity
shrinks ~55% but stays significant (t −2.76).
Verdict changes: not yet computed (Phase 3); the composite-H2 test is at risk.
Bugs fixed: FY labels; removed LM words; multi-class shares; build_text_features
crash on empty section-source columns; SIC fallback.
Limitations added: #26 (exhibit share not measured); ~15 FY2024 firm-years
without text after the FY fix; 6 firms without Yahoo prices; universe
look-ahead (snapshot 2026-10-12).
Open questions for owner: see chat (FF48 map = new data source?; idio-vol dof).
Files changed: scripts/{sec_edgar,build_annual_panel,config,panel,
build_volatility_horizons,backfill_filing_metadata,build_text_features,
text_features,lm_loader,check_accessions,universe_attrition}.py, docs/*.

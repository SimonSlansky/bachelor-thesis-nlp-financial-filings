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
  almost no variation at 30d.

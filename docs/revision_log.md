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

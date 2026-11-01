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

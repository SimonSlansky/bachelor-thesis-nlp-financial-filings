# Item 1A Risk-Factor Disclosure and Post-Filing Volatility

Bachelor thesis on whether the volume, wording (Loughran–McDonald uncertainty and litigious density) and peer-relative omissions of 10-K Item 1A risk-factor disclosure predict post-filing stock-return volatility; 548 large U.S. non-financial firms, fiscal years 2010–2024.

## Repository Structure

```
├── data/                # Raw and processed datasets (CSV, gitignored)
├── docs/                # Thesis plan, timeline, literature articles & abstracts
├── scripts/             # Data collection & panel construction pipelines
├── tex/                 # LaTeX thesis source (VŠE template)
│   ├── prace.tex        # Main document
│   ├── makra.tex        # Packages & formatting
│   ├── zacatek.tex      # Title page, abstract, AI declaration
│   ├── uvod.tex         # Introduction
│   ├── chapters/        # 02_literature_review … 06_robustness
│   ├── zaver.tex        # Conclusion
│   ├── literatura.tex   # Bibliography
│   ├── zkratky.tex      # Abbreviations
│   ├── app01.tex        # Appendix A
│   ├── app02.tex        # Appendix B (AI usage)
│   ├── tables/          # Script-generated LaTeX tables
│   └── bibliography.bib
├── requirements.txt
└── .gitignore
```

## Data Pipeline

**Frozen inputs — never re-download or regenerate:** `data/text_sections.csv`
(Item 1A / Item 7 text, extracted once in Oct 2026; `text_parser.py` is kept
for documentation only), `data/text_accessions.csv`, `data/lm_master.csv`,
`data/universe.csv`, `data/cik_map.csv`, `data/prices/*.csv` (price cache:
files are only added, never refreshed), `data/validation_sample.csv`,
`data/extraction_validation.csv`, `data/external/Siccodes48.txt`.
Variable definitions: `docs/data_dictionary.md`; revision history and
decisions: `docs/revision_log.md`.

Run from `scripts/` with a venv **outside** cloud-synced folders
(e.g. `~/.venvs/bakalarka`, `pip install -r requirements.txt`):

1. Data rebuild (`zsh phase1_run.sh`, ~1–2 h; SEC JSON + header pages, Yahoo):
   `build_annual_panel.py --skip-text` → `fetch_market_data.py` →
   `fetch_earnings_dates.py` → `build_volatility_horizons.py` →
   `backfill_filing_metadata.py` → `build_text_features.py`;
   then `check_accessions.py`, `universe_attrition.py`
2. `build_silence.py` — H3 omission measures (~15 min)
3. `regressions.py` — H1/H2 models, robustness, wild bootstrap → `data/results/`
4. `silence_table.py`, `silence_robust_table.py`, `silence_placebo.py --n 1000`
5. `stationarity.py`
6. Validation: `validation_table.py`, `validation_second_llm.py`,
   `validation_agreement.py` (needs `admin/human_labels_template.csv` labels)
7. `make_tables.py`, `make_figures.py`
8. `check_numbers.py` — every number in the chapters must match a result (0 unmatched)

## Building the Thesis

```bash
cd tex
latexmk          # non-interactive; build files in /tmp/latex_build_bakalarka
```

## Requirements

- Python 3.11+ (`requirements.txt`)
- LaTeX distribution with `biber` (e.g., TeX Live, MiKTeX)

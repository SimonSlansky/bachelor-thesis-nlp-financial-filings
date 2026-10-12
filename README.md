# Textual-Financial Divergence and Stock Return Volatility

Bachelor thesis investigating whether the divergence between narrative change and financial change in SEC filings predicts future stock return volatility, using a panel of ~540 large U.S. non-financial firms over 2010–2024.

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

1. **SEC EDGAR XBRL** → `scripts/build_annual_panel.py`
2. Shared modules: `config.py` (constants), `sec_edgar.py` (API), `returns.py` (yfinance), `panel.py` (ratios & imputation)
3. Raw output → `data/annual_financials.csv`
4. Final panel → `data/annual_panel.csv`

## Building the Thesis

```bash
cd tex
latexmk -pdf prace.tex
```

## Requirements

- Python 3.12+
- `pip install -r requirements.txt`
- LaTeX distribution with `biber` (e.g., TeX Live, MiKTeX)

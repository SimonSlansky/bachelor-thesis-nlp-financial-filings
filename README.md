# Textual-Financial Divergence and Stock Return Volatility

Bachelor thesis investigating whether the divergence between narrative change and financial change in SEC filings predicts future stock return volatility, using a panel of ~600 large U.S. non-financial firms over 2010–2025.

## Repository Structure

```
├── data/                # Raw and processed datasets (CSV, gitignored)
├── docs/                # Thesis plan, timeline, literature articles & abstracts
├── output/              # Script-generated figures and tables
│   ├── figures/
│   └── tables/
├── scripts/             # Data collection & panel construction pipelines
├── tex/                 # LaTeX thesis source
│   ├── main.tex
│   ├── chapters/        # 01_introduction … 07_conclusion
│   ├── frontmatter/     # Title page, abstract, declaration, acknowledgments
│   ├── backmatter/      # Appendix
│   └── bibliography.bib
├── requirements.txt
└── .gitignore
```

## Data Pipeline

1. **SEC EDGAR XBRL** → `scripts/build_annual_panel.py` / `scripts/build_quarterly_panel.py`
2. Shared modules: `config.py` (constants), `sec_edgar.py` (API), `returns.py` (yfinance), `panel.py` (ratios & imputation)
3. Raw output → `data/annual_financials.csv`, `data/quarterly_financials.csv`
4. Final panels → `data/annual_panel.csv`, `data/quarterly_panel.csv`

## Building the Thesis

```bash
cd tex
latexmk -pdf main.tex
```

## Requirements

- Python 3.12+
- `pip install -r requirements.txt`
- LaTeX distribution with `biber` (e.g., TeX Live, MiKTeX)

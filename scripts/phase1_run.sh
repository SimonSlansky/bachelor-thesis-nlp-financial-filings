#!/bin/zsh
# Phase-1 rebuild. Run from repo root:  zsh scripts/phase1_run.sh > admin/phase1_run.log 2>&1 &
set -e
export PYTHONUNBUFFERED=1
PY=~/.venvs/bakalarka/bin/python
cd "$(dirname "$0")"
echo "== $(date) 1/7 XBRL re-extraction (point-in-time, consolidated NI, shares)"
$PY build_annual_panel.py --skip-text
echo "== $(date) 2/7 market data (SPY, as-traded prices + splits)"
$PY fetch_market_data.py
echo "== $(date) 3/7 earnings-announcement dates (8-K Item 2.02)"
$PY fetch_earnings_dates.py
echo "== $(date) 4/7 volatility horizons + market controls"
$PY build_volatility_horizons.py
echo "== $(date) 5/7 SKIPPED: 10-K text is never re-downloaded (owner decision)"
echo "== $(date) 6/7 historical SIC back-fill (header pages only, no 10-K download)"
$PY backfill_filing_metadata.py
echo "== $(date) 7/7 text features + merge"
$PY build_text_features.py
echo "== $(date) DONE"

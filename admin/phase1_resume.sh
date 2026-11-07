#!/bin/zsh
# Resume of phase1_run.sh after the FY-label and shares fixes (see revision_log).
# Step 6 (SIC headers) runs in parallel: admin/phase1_sic.log
set -e
export PYTHONUNBUFFERED=1
PY=~/.venvs/bakalarka/bin/python
cd "$(dirname "$0")/../scripts"
echo "== $(date) 1/7 XBRL re-extraction (FY from period end, shares fallback)"
$PY build_annual_panel.py --skip-text
echo "== $(date) 4/7 volatility horizons + market controls"
$PY build_volatility_horizons.py
echo "== $(date) waiting for SIC back-fill"
while pgrep -f backfill_filing_metadata.py > /dev/null; do sleep 20; done
$PY backfill_filing_metadata.py
echo "== $(date) 7/7 text features + merge"
$PY build_text_features.py
echo "== $(date) DONE"

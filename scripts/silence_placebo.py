"""Placebo test for the H3 strategic-silence result (peer-cell shuffle).

Within each fiscal year, firms are randomly reassigned to SIC2 cells while
keeping cell sizes fixed. The omission gap is recomputed against these fake
peers and the baseline firm + year FE model at h = 30 days is re-estimated.
If the real result reflects silence on peer-relevant risks, the placebo
coefficients should be close to zero.

Writes data/silence_placebo.csv (one row per permutation).
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_silence import silence_for_section, load_sections
from silence_table import fit

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
N_PERM = 5
SEED = 20260101


def main() -> None:
    sec = load_sections(cols=("item_1a",)).reset_index(drop=True)
    panel = pd.read_csv(DATA / "annual_panel_text.csv")

    rng = np.random.default_rng(SEED)
    rows = []
    firm_sic = sec.groupby("ticker")["sic"].first()
    for k in range(1, N_PERM + 1):
        fake = sec.copy()
        # Shuffle SIC codes across FIRMS (fixed over time), so the peer set
        # of years t and t-1 stays consistent but is economically random.
        perm = pd.Series(rng.permutation(firm_sic.to_numpy()),
                         index=firm_sic.index)
        fake["sic"] = fake["ticker"].map(perm)
        om = silence_for_section(fake, "item_1a", "1a")[["ticker", "fiscal_year", "omission_1a"]]
        df = panel.merge(om, on=["ticker", "fiscal_year"], how="left")
        r = fit(df, 30, True)   # real SIC still used for nothing here (firm FE)
        rows.append({"perm": k, **r})
        print(f"  permutation {k}: b={r['b']:+.3f}  t={r['t']:+.2f}  n={r['n']}")

    out = pd.DataFrame(rows)
    out.to_csv(DATA / "silence_placebo.csv", index=False)
    print(f"\nmax |t| = {out['t'].abs().max():.2f}")


if __name__ == "__main__":
    main()

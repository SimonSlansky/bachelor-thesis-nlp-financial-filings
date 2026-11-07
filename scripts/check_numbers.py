"""Check that every number in the thesis prose comes from a generated output.

Extracts numbers from tex/chapters/*.tex, uvod.tex, zaver.tex, app01.tex and
app02.tex (comments, labels, refs and citations removed) and matches each
against the pool of values in data/results/*.csv, data/results/*.json, the
Phase-1 diagnostic CSVs, data/universe_stages.csv and the generated
tex/tables/*.tex.  A number with d decimals matches if some source value
rounds to it at d decimals (also as a percentage, x100).  Years, small
integers (<= 20), design constants and literature numbers listed in
admin/check_numbers_allow.txt are skipped.

Prints unmatched numbers with context; exit code 1 if any.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TEXFILES = sorted((ROOT / "tex" / "chapters").glob("*.tex")) + [
    ROOT / "tex" / f for f in ("uvod.tex", "zaver.tex", "app01.tex", "app02.tex")]
DIAG = ["tag_overlap.csv", "tag_locking_summary.csv", "sample_attrition.csv",
        "text_coverage_by_year.csv", "min_years_sensitivity.csv", "universe_stages.csv",
        "accession_changes.csv"]
CONSTANTS = {252, 200, 100, 60, 90, 180, 365, 1000, 3000, 9999, 2000, 300, 500, 548,
             0.95, 0.05, 0.01, 0.10, 0.1, 99, 30, 40, 50, 1.96}
NUM = re.compile(r"(?<![\w.\\])[-−+]?\d{1,3}(?:[,{}]*\d{3})*(?:\.\d+)?|(?<![\w.\\])[-−+]?\d+\.\d+")


def clean_tex(s: str) -> str:
    s = re.sub(r"(?<!\\)%.*", "", s)
    s = re.sub(r"\\(label|ref|eqref|cite|textcite|parencite|input|include|includegraphics|"
               r"autoref|cref|url|href)\*?(\[[^\]]*\])*\{[^}]*\}", " ", s)
    s = re.sub(r"\\(begin|end)\{[^}]*\}", " ", s)
    s = re.sub(r"\[[0-9.]+(pt|cm|mm|em|ex)\]|[0-9.]+\\textwidth|[0-9.]+\\linewidth", " ", s)
    return s.replace("{,}", ",").replace("\\,", "")


def source_pool():
    vals = []
    files = list((ROOT / "data" / "results").glob("*.csv")) + [ROOT / "data" / f for f in DIAG]
    for f in files:
        if not f.exists():
            continue
        try:
            d = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        for c in d.columns:
            vals += pd.to_numeric(d[c], errors="coerce").dropna().tolist()
            vals += pd.to_numeric(pd.Series([c]), errors="coerce").dropna().tolist()
    for f in (ROOT / "data" / "results").glob("*.json"):
        vals += [float(x) for x in re.findall(r"-?\d+\.?\d*(?:e-?\d+)?", f.read_text())]
    for f in (ROOT / "tex" / "tables").glob("*.tex"):
        for m in NUM.findall(clean_tex(f.read_text(encoding="utf-8"))):
            vals.append(to_float(m))
    pool = {d: set() for d in range(0, 6)}
    for v in vals:
        if v is None or v != v:
            continue
        for x in (abs(v), abs(v) * 100, v, v * 100):
            for d in pool:
                pool[d].add(round(x, d))
    return pool


def to_float(tok: str):
    t = tok.replace("−", "-").replace(",", "").replace("{", "").replace("}", "")
    try:
        return float(t)
    except ValueError:
        return None


def allowlist():
    p = ROOT / "admin" / "check_numbers_allow.txt"
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            out.add(line)
    return out


def main():
    pool = source_pool()
    allow = allowlist()
    bad = []
    for f in TEXFILES:
        if not f.exists():
            continue
        for ln, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            s = clean_tex(line)
            for m in NUM.finditer(s):
                tok = m.group(0)
                x = to_float(tok)
                if x is None:
                    continue
                ax = abs(x)
                dec = len(tok.split(".")[1]) if "." in tok else 0
                if dec == 0 and (ax <= 20 or 1900 <= ax <= 2100 or ax in CONSTANTS):
                    continue
                if ax in CONSTANTS or tok.lstrip("+-−") in allow:
                    continue
                signed = tok[0] in "+-−" and m.start() > 0 and s[m.start() - 1] in "$( ="
                key = round(x if signed else ax, min(dec, 5))
                if key in pool[min(dec, 5)]:
                    continue
                bad.append((f.relative_to(ROOT), ln, tok, s.strip()[:110]))
    for f, ln, tok, ctx in bad:
        print(f"{f}:{ln}: {tok:>10}   {ctx}")
    print(f"\n{len(bad)} unmatched numbers")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

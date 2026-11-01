"""Verify bibliography entries against the Crossref REST API (Phase 6).

For every entry in tex/bibliography.bib plus the CANDIDATES below (references
the revision plans to add), query Crossref (by DOI if known, else by
bibliographic query) and record the best match: DOI, title, authors, journal,
year, volume, issue, pages, and a similarity score against the stored title.
Nothing is written to the .bib automatically.

Writes admin/crossref_check.csv.  Network: api.crossref.org (JSON) only.
"""
from __future__ import annotations

import difflib
import re
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
BIB = ROOT / "tex" / "bibliography.bib"
OUT = ROOT / "admin" / "crossref_check.csv"
UA = {"User-Agent": "thesis-bib-check (mailto:simon.slansky@outlook.com)"}

CANDIDATES = [  # key, query (author + title words + journal), expected year
    ("loughran2016", "Loughran McDonald|Textual analysis in accounting and finance: a survey", 2016),
    ("kogan2009", "Kogan Levin Routledge Sagi Smith|Predicting risk from financial reports with regression", 2009),
    ("filzen2015", "Filzen|The information content of risk factor disclosures in quarterly reports", 2015),
    ("bao2014", "Bao Datta|Simultaneously discovering and quantifying risk types from textual risk disclosures", 2014),
    ("beatty2019", "Beatty Cheng Zhang|Are risk factor disclosures still relevant? market reactions financial crisis", 2019),
    ("verrecchia1983", "Verrecchia|Discretionary disclosure", 1983),
    ("dye1985", "Dye|Disclosure of nonproprietary information", 1985),
    ("kothari2009withhold", "Kothari Shu Wysocki|Do managers withhold bad news", 2009),
    ("cameron2015", "Cameron Miller|A practitioner's guide to cluster-robust inference", 2015),
    ("nickell1981", "Nickell|Biases in dynamic models with fixed effects", 1981),
    ("kronmal1993", "Kronmal|Spurious correlation and the fallacy of the ratio standard revisited", 1993),
    ("hoberg2016", "Hoberg Phillips|Text-based network industries and endogenous product differentiation", 2016),
    ("harris1999", "Harris Tzavalis|Inference for unit roots in dynamic panels where the time dimension is fixed", 1999),
    ("ips2003", "Im Pesaran Shin|Testing for unit roots in heterogeneous panels", 2003),
    ("holm1979", "Holm|A simple sequentially rejective multiple test procedure", 1979),
    ("cameron2008", "Cameron Gelbach Miller|Bootstrap-based improvements for inference with clustered errors", 2008),
    ("webb2023", "Webb|Reworking wild bootstrap-based inference for clustered errors", 2023),
    ("cohen1960", "Cohen|A coefficient of agreement for nominal scales", 1960),
    ("wilson1927", "Wilson|Probable inference, the law of succession, and statistical inference", 1927),
    ("ang2006", "Ang Hodrick Xing Zhang|The cross-section of volatility and expected returns", 2006),
    ("arellano1991", "Arellano Bond|Some tests of specification for panel data Monte Carlo evidence", 1991),
    ("dickeyfuller1979", "Dickey Fuller|Distribution of the estimators for autoregressive time series with a unit root", 1979),
    ("maddala1999", "Maddala Wu|A comparative study of unit root tests with panel data and a new simple test", 1999),
    ("mackinnon2017", "MacKinnon Webb|Wild bootstrap inference for wildly different cluster sizes", 2017),
    ("roodman2019", "Roodman Nielsen MacKinnon Webb|Fast and wild: bootstrap inference in Stata using boottest", 2019),
    ("heinle2018", "Heinle Smith Verrecchia|Risk-factor disclosure and asset prices", 2018),
    ("lang2015", "Lang Stice-Lawrence|Textual analysis and international financial reporting", 2015),
    ("gaulin2017", "Gaulin|Risk fact or fiction: the information content of risk factor disclosures", 2017),
]


def parse_bib(text: str) -> list[dict]:
    out = []
    for m in re.finditer(r"@(\w+)\{([^,]+),(.*?)\n\}", text, re.S):
        typ, key, body = m.groups()
        f = {k.lower(): v.strip() for k, v in
             re.findall(r"(\w+)\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body)}
        out.append({"key": key.strip(), "type": typ, **f})
    return out


def clean(s: str) -> str:
    return re.sub(r"[{}\\\"'`]", "", s or "").lower()


def crossref(query: str | None = None, doi: str | None = None,
             title: str | None = None) -> dict | None:
    """DOI lookup, else bibliographic query; journal articles are preferred
    and ranked by title similarity to *title*."""
    for k in range(3):
        try:
            if doi:
                r = requests.get(f"https://api.crossref.org/works/{doi}", headers=UA, timeout=20)
                if r.status_code == 200:
                    return r.json()["message"]
                if r.status_code == 404:
                    return None
            else:
                items = []
                for flt in ("type:journal-article", None):
                    params = {"query.bibliographic": query, "rows": 5}
                    if flt:
                        params["filter"] = flt
                    r = requests.get("https://api.crossref.org/works", headers=UA,
                                     timeout=20, params=params)
                    if r.status_code == 200:
                        items += r.json()["message"]["items"]
                    time.sleep(0.3)
                if not items:
                    return None
                if title:
                    def sim(it):
                        t = (it.get("title") or [""])[0]
                        return difflib.SequenceMatcher(None, clean(title), clean(t)).ratio()
                    best = max(sim(it) for it in items)
                    good = [it for it in items if sim(it) >= best - 0.02]
                    jr = [it for it in good if it.get("type") == "journal-article"
                          and "ssrn" not in (it.get("DOI") or "").lower()]
                    return (jr or good)[0]
                return items[0]
        except Exception:
            pass
        time.sleep(2 ** k)
    return None


def row(key, stored_title, stored_year, it):
    if not it:
        return {"key": key, "found": False}
    title = (it.get("title") or [""])[0]
    year = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
    authors = "; ".join(f"{a.get('family', '')}, {a.get('given', '')}" for a in it.get("author", []))
    sim = difflib.SequenceMatcher(None, clean(stored_title), clean(title)).ratio() if stored_title else None
    return {"key": key, "found": True, "doi": it.get("DOI"), "cr_title": title,
            "cr_authors": authors, "cr_journal": (it.get("container-title") or [""])[0],
            "cr_year": year, "cr_volume": it.get("volume"), "cr_issue": it.get("issue"),
            "cr_pages": it.get("page"), "cr_type": it.get("type"),
            "stored_title": stored_title, "stored_year": stored_year, "title_similarity": sim}


def main():
    rows = []
    for e in parse_bib(BIB.read_text(encoding="utf-8")):
        q = f"{clean(e.get('author', ''))} {clean(e.get('title', ''))} {clean(e.get('journal', ''))}"
        it = crossref(doi=e["doi"]) if e.get("doi") else crossref(query=q, title=e.get("title"))
        rows.append({"source": "bib", **row(e["key"], e.get("title"), e.get("year"), it),
                     "stored_journal": e.get("journal"), "stored_volume": e.get("volume"),
                     "stored_pages": e.get("pages")})
        print(f"  bib  {e['key']:22s} {rows[-1].get('title_similarity')}", flush=True)
        time.sleep(0.3)
    for key, q, yr in CANDIDATES:
        authors, title = q.split("|")
        it = crossref(query=f"{authors} {title}", title=title)
        rows.append({"source": "candidate", **row(key, title, yr, it)})
        print(f"  cand {key:22s} {rows[-1].get('cr_title')!r} {rows[-1].get('cr_year')}", flush=True)
        time.sleep(0.3)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()

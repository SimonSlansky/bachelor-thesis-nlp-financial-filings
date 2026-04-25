# Přehled pro vedoucího – Data & Results

**Práce:** Textual-Financial Divergence and Stock Return Volatility: Evidence from U.S. Corporate Filings
**Autor:** Šimon Slanský | **Vedoucí:** prof. RNDr. Ing. Michal Černý, Ph.D.
**Stav k:** duben 2026

---

## Datová část (hotovo)

### Vzorek
- Začal jsem s top 1 000 amerických firem podle tržní kapitalizace ze SEC EDGAR.
- Filtry:
  1. Jen *operating* entity → vyřazení ETF, fondů, shell companies.
  2. Vyloučení finančního sektoru SIC 6000–6999 (banky, pojišťovny, REIT) — jejich účetní výkazy nejsou srovnatelné s průmyslovými firmami, standardní omezení v empirické účetní literatuře (Sloan 1996, Francis et al. 2005).
  3. Minimum 5 let strojově čitelných XBRL dat z 10-K.
  4. ≥90 % pokrytí akciových výnosů (trading days v měřicím okně).
- Výsledný vzorek: **539 firem, 6 322 pozorování, FY 2010–2024**.
- Nevyvážený panel (ne všechny firmy mají všechny roky) — odpovídá standardu v literatuře.

### Finanční data z XBRL
- Roční data přímo z SEC EDGAR Company Facts API (10-K filings).
- Extrahované metriky: total assets, total liabilities, net income, operating income, operating cash flow, stockholders' equity, assets/liabilities current.
- Problém: stejný účetní koncept se v XBRL může skrývat pod různými tagy (např. operating cash flow má tag pro *total activities* i *continuing operations only*). Řešení dvoustupňovou harmonizací:
  - *Ekvivalentní tagy* (jen jiná verze taxonomie, např. dva tagy pro OCF) → nechávám koexistovat v rámci firmy. Empirická kontrola: na 1 352 překryvech OCF byl medián rozdílu 0.000 pp, na 4 761 překryvech NI byl medián ROA bias 0.033 pp.
  - *Ekonomicky odlišné tagy* (např. equity s/bez NCI) → per-firm tag locking: pro každou firmu vyberu tag, který firma používá nejčastěji, a hodnoty z ostatních tagů nastavím na NaN.
- Výsledek: complete-case rate pro 6 regresních kontrol stoupla z 78,9 % na 92,7 %. Diagnostika tagů je v příloze práce (Table A.2).
- Imputace: `total_liabilities = total_assets − stockholders_equity` kde chybí; ±1 periodový ffill/bfill na balance-sheet položky.

### Akciová volatilita
- Zdroj: yfinance (adjusted close prices).
- Anualizovaná volatilita = std(denní log-výnosy) × √252.
- Okno: 365 kalendářních dní, začíná 2 obchodní dny po filing date 10-K.
  - Proč post-filing? Zarovnáváme informační šok na datum zveřejnění (event-study logika).
  - 2denní lag: filing se typicky zpracovává trhem s mírným zpožděním.
- Minimum 200 obchodních dní v okně, jinak NaN.
- Firmy s <90 % non-NaN volatilitou (po odfiltrování lag-roku a cap FY ≤ 2024) vyřazeny → zajišťuje, že regresní vzorek nemá systematické mezery.

### Finanční poměrové ukazatele
- Odvozeno z raw XBRL metrik:
  - **log(total assets)** — proxy pro velikost firmy.
  - **leverage** = total liabilities / total assets.
  - **ROA** = net income / total assets.
  - **asset growth** = (TA_t − TA_{t-1}) / TA_{t-1}, s ochranou proti mezerám způsobeným změnou FYE (validní jen pro 300–400denní gap).
  - **current ratio** = current assets / current liabilities.
  - **OCF-to-assets** = OCF / total assets.
- Winzorizace na 1./99. percentilu u: leverage, ROA, asset growth, current ratio, OCF-to-assets.
- Nejstarší rok firmy odstraněn (spotřebován lag pro asset growth).

### Extrakce textu z 10-K
- Cíl: vytáhnout z každého 10-K čistý text Item 1A (Risk Factors) a Item 7 (MD&A).
- Problém: sekce nemají standardní tagování; hlavičky jsou různě formátované (CSS bold, `<font>`, uppercase, tabulkové layouty), smíchané s obsahem (TOC), page headers a inline cross-referencemi.
- Řešení: vlastní HTML-DOM parser se 4 fázemi:
  1. **Header detection** — procházím DOM na úrovni bloků, testuji regex pro „Item 1A" / „Item 7", filtruji jen elementy s bold/uppercase/large-font; odstraňuji TOC linky a running headers.
  2. **Position marking** — do DOM vkládám unikátní markery, které po flatten na plain text dávají pozice hlaviček.
  3. **Candidate selection** — subtitle keyword check (300 znaků), TOC-cluster pravidlo (≥3 sousední hlavičky = TOC), cross-reference filtr; vybírám kandidáta s nejdelším textem.
  4. **Boundary detection** — sekce končí u dalšího Item header; sekce <200 slov → exhibit fallback (stáhnu exhibit z filing indexu).
- Post-processing: odstranění page headers/footers, standalone page numbers, `&nbsp;` → mezera, garbled tabulkové řádky.
- Výsledek: ~6 500 zpracovaných filingů (10-K/A amendments vynechány — obsahují jen cover page).
- **AI validace** (nezávislý audit):
  - Nástroj: Google Gemini 2.5 Flash (temperature = 0, deterministic).
  - Vzorek: 200 sekcí (100 × Item 1A, 100 × Item 7), stratifikováno po 5 érách (2010–2024).
  - 5 kritérií: správná sekce, start boundary, end boundary, kvalita obsahu, kompletnost.
  - Výsledek: **93,5 % lenient pass rate** (Pass + Minor), 95% CI: 89,2–96,2 %.
  - Selhání se koncentrují v raných letech (5/40 v 2010–2012 vs. 1/40 v 2022–2024) → artefakt nestandardního HTML, nikoli systematický problém.

### Proč roční a ne kvartální data
Kvartální data by ztrojnásobila vzorek, ale tři problémy to znemožnily:
1. **Chybějící risk faktory v 10-Q** — SEC vyžaduje aktualizaci jen při „material changes"; v pilotním vzorku (15 firem, 90 filingů) mělo substantivní risk-factor sekci jen 52 % 10-Q. Nemožnost konstruovat ΔRisk.
2. **Nespolehlivá Q4 finanční data** — 10-Q pokrývá jen Q1–Q3; Q4 se musí dopočítat jako Annual − Q1 − Q2 − Q3. OCF pokrytí jen 7 % firm-years, ~11 % mismatch kde oba údaje existují → cash-flow odvozené ratios na kvartálních datech nepoužitelné.
3. **Konsistence s literaturou** — Loughran & McDonald (2011), Li (2008), Campbell et al. (2014) pracují s 10-K.
- Zdokumentováno diagnostickými tabulkami v příloze (XBRL coverage, Q4 reliability, text availability).

---

## Výsledky – co už je hotovo

### 1. Deskriptivní statistiky (Table 4 v práci)

| Proměnná | Průměr | SD | Medián |
|---|---|---|---|
| Volatilita (t+1) | 0.325 | 0.168 | 0.281 |
| Zpožděná volatilita | 0.324 | 0.171 | 0.279 |
| ln(Total Assets) | 23.1 | 1.48 | 23.2 |
| Leverage | 0.61 | 0.22 | 0.61 |
| ROA | 0.062 | 0.087 | 0.061 |
| Asset Growth | 0.119 | 0.254 | 0.060 |

- **537 firem, 6\,299 pozorování.**
- Volatilita má pravostranné rozdělení (IQR 0.22–0.39), konzistentní s občasnými výkyvy (covid 2020).

### 2. Korelační matice (Table 5)

Klíčové poznatky:
- **Volatilita vs. zpožděná vol: r = 0.61** → silná persistence.
- **Velikost vs. vol: r = −0.32** → větší firmy nižší vol (klasický vztah).
- **ROA vs. vol: r = −0.32** → ziskovější firmy nižší vol.
- Žádná korelace mezi explanačními proměnnými nepřekračuje |0.40| → VIF < 1.4, nízké riziko multikolinearity.

### 3. Baseline regrese – finanční model (Table 6)

Tři nested specifikace, všechny s odvětvovými (SIC-2, 49 skupin) a ročními FE, SE clusterované po firmě:

**Model (1): Jen zpožděná volatilita**
- Koef. 0.67, t = 33.5 → silná persistence.
- Adj. R² = 0.630.
- Interpretace: jeden procentní bod nárůst loňské vol → ~0.67 pp nárůst letošní vol.

**Model (2): + velikost a leverage**
- Velikost: −0.017, t = −7.4 → větší firmy = nižší budoucí vol.
- Leverage: ~0, nesignifikantní (t = 0.17) → industry FE už absorbují průřezovou variaci v kapitálové struktuře.
- Adj. R² = 0.643 (+1.3 pp).

**Model (3): + ROA, asset growth (plný baseline)**
- **ROA: −0.224, t = −7.9** → silný negativní efekt, ziskovější firmy mají nižší budoucí vol.
- **Asset growth: +0.026, t = 3.1** → rychlý růst = vyšší nejistota.
- **Adj. R² = 0.654** (+1.1 pp oproti modelu 2).

### Co to znamená

Finanční baseline vysvětluje **65,4 %** variace v budoucí volatilitě. To je vysoký benchmark. Hlavní drivery jsou:
1. **Persistence** (zpožděná vol) – suverénně nejsilnější prediktor; koef. klesá z 0.67 na 0.55 jak přidáváme kontroly, ale zůstává dominantní.
2. **Velikost** – menší firmy volatilnější; koef. stabilní napříč modely (−0.017).
3. **Ziskovost (ROA)** – neziskové firmy volatilnější; ekonomicky velký efekt (−0.22).
4. **Růst aktiv** – expanze = vyšší nejistota; menší efekt ale jasně signifikantní.
5. **Leverage** – nesignifikantní po zahrnutí FE a ostatních kontrol.

Jakékoliv textové proměnné musí přinést inkrementální vysvětlující sílu nad tímto už poměrně silným modelem. I malý nárůst adj. R² (řádově 0.5–2 pp) je ekonomicky významný, protože baseline je už nasycený persistence + FE.

---

## Co zbývá udělat (další fáze)

### Textové features (fáze 3)
1. **LM dictionary sentiment** – Loughran–McDonald slovník specifický pro finanční texty. Počítám % negativních, pozitivních a uncertainty slov v Item 7 (MD&A). Důvod: obecné slovníky (Harvard GI) špatně fungují ve finance, protože slova jako „liability" mají jinou konotaci.
2. **Risk-word density** – uncertainty + litigious + constraining slova v Item 1A (Risk Factors) / celkový počet slov.
3. **TF-IDF + cosine similarity** – vektorizuji každý dokument, porovnám s minulým rokem téže firmy. Vysoká cosine similarity = firma málo změnila text; nízká = nová informace.
4. **Risk factor diff** – jednotlivé risk faktory se dají separovat (sub-headers v Item 1A). Pro každý rok a firmu spočítám: kolik faktorů bylo přidáno, odebráno a modifikováno. Silný prediktor nejistoty (Lyle, Riedl & Siano 2023).
5. **FinBERT** – transformer model (ProsusAI/finbert) trénovaný na finančních textech. Scoring na úrovni vět → P(positive), P(negative), P(neutral) → agregace průměrem přes celou sekci. Výhoda: chápe kontext, negaci, na rozdíl od bag-of-words.
   - ~6 300 filingů × ~600 vět = ~3.8M sentence inferences. GPU ideální, CPU proveditelné (~6–10 hodin).

### Divergence (fáze 3)
- **Hlavní míra**: residuál z regrese ΔSentiment na ΔROA. Interpretace: jak moc se změnil tón managementu nad rámec toho, co bychom čekali z finanční výkonnosti.
- **Binární alternativa**: DivDummy = 1 pokud se sentiment zlepšil ale ROA zhoršila (nebo naopak).
- Winzorizace na 1./99. percentilu po konstrukci.

### Regrese (fáze 4)
Všechny modely: OLS s absorbovanými SIC-2 + fiscal-year FE, SE clusterované po firmě (Petersen 2009).

- **H1 (text přidává info):** Text model = baseline + ΔSentiment, ΔRisk density, ΔTextSim, risk factor turnover, ΔFinBERT.
  - Testováno: joint F-test na textové koeficienty; inkrementální adj. R² nad baseline.
- **H2 (divergence je nový signál):** Divergence model = text model + Divergence.
  - Testováno: t-test na koeficient divergence; inkrementální adj. R² nad text model.
  - Toto je klíčový výsledek celé práce.

### Robustnost (fáze 5)
- Alternativní vol okna (63d, 180d místo 365d).
- Post-filing log returns jako alternativní závislá proměnná.
- Binární divergence místo continuous.
- Alternativní ΔROA (operating ROA, OCF-to-assets).
- Subsample: pre-2020 vs. post-2020; tech vs. non-tech.
- Firma FE místo industry FE.
- Winzorizace 2.5/97.5 místo 1/99.

### Psaní (fáze 6)
- Metodologie → Zbylé Results (H1/H2) → Robustnost → Literature review → Úvod → Závěr.

---

## Nejbližší kroky

1. Stáhnout Loughran–McDonald slovník a napsat skript na dictionary features (sentiment, uncertainty, litigious).
2. TF-IDF vektorizace + cosine similarity mezi po sobě jdoucími roky pro každou firmu.
3. FinBERT scoring (~3.8M vět, ideálně na GPU).
4. Merge textových features do panelu → `annual_panel_full.csv`.
5. Zkonstruovat divergenci (residuál ΔSent ~ ΔROA) a binární variantu.
6. Odhadnout text model (H1) a divergenční model (H2) → porovnat adj. R² s baseline (0.654).
7. Dopsat Metodologii a Results.

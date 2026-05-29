# Průvodce bakalářskou prací — podklady k obhajobě

> **Téma:** Dekompozice jazyka rizikových faktorů v 10-K výkazech a následná volatilita akcií.
> **Klasifikace práce:** prediktivní (ne kauzální) studie kombinující text mining 10-K filingů s panelovou regresí.
> **Vzorek:** 535 nefinančních firem (top 1000 US dle market capu), 2010–2024, headline horizont 30 dní po filingu.

Každá sekce odpovídá na: **CO** jsi udělal, **PROČ** tak, **JAK** technicky a **CO** z toho vyšlo.

---

## 0. Velký obrázek (one-pager)

- **Problém:** dvě literatury si protiřečí.
  - Campbell et al. (2014): delší Item 1A → vyšší následná volatilita („volume channel").
  - Hope et al. (2016): specifičtější text → nižší volatilita („specificity channel").
- **Cíl:** oddělit oba kanály v jedné regresi a přidat třetí — co firma vynechá vůči peerům (H3).
- **3 hypotézy:**
  - **H1** (objem): β_len > 0 — delší text → vyšší σ. **Potvrzeno** (+0,121; t = +7,64).
  - **H2** (specifičnost): β_dens < 0 při kontrole délky. **Potvrzeno** (−3,88; t = −2,68); v dekompozici nese efekt **LM-Litigious** (−6,16), nikoli LM-Uncertainty.
  - **H3** (mlčení): β_om > 0 ve firm-FE specifikaci. **Potvrzeno within-firm** (+1,345; t = +2,93), nulové v cross-sectionu.
- **Hlavní přínos:** modulární pipeline (SEC EDGAR → XBRL tag locking → HTML extrakce → LM dekompozice → peer-omission konstrukt) ověřená nezávislým LLM auditem (93,5 %).

---

## 1. Úvod (`uvod.tex`)

- Motivuje rozpor mezi Campbell (volume) a Hope (specificity) — dva opačné znaménka efektu růstu Item 1A.
- Vymezuje práci jako **prediktivní** (ne kauzální) — záměrně se vyhýbá silným kauzálním tvrzením.

> **Argumentace u obhajoby:** „Práce se neptá, zda jazyk *způsobuje* volatilitu, ale zda ji *informativně předpovídá* nad rámec finančních kontrol."

---

## 2. Literární rešerše (`02_literature_review.tex`)

Klíčové reference, které musíš znát a používat v argumentaci:

| Autor | Co dokazuje | Kde v práci |
|---|---|---|
| **Campbell et al. (2014)** | Delší Item 1A → vyšší σ. „Volume channel". | H1, motivace |
| **Hope et al. (2016)** | Specifičtější text → nižší σ. „Specificity channel". | H2, motivace |
| **Loughran & McDonald (2011)** | Finanční sentiment slovník (LM). Používám sub-listy Uncertainty a Litigious. | H2 dekompozice |
| **Nelson & Pritchard (2007)** | Safe-harbour → firmy přidávají právní boilerplate. | Vysvětluje, proč LM-Litigious nese efekt |
| **Cazier et al. (2021), Beatty et al. (2019)** | Risk factory se rok od roku recyklují → firm-level trvalost. | Vysvětluje, proč firm-FE pohřbí LM efekty |
| **Brown & Tucker (2011), Dyer et al. (2017)** | Disclosure-similarity literatura, SIC2 jako benchmark. | H3 konstrukce |
| **Kravet & Muslu (2013)** | Risk-disclosure preferuje likviditní proxy nad růstovými. | H3 kontroly |
| **Petersen (2009), Cameron et al. (2011)** | Dvouvrstvové clusterování SE (firma × rok). | Estimation |
| **Gilardi et al. (2023)** | LLM jako nezávislý validátor textových úloh. | LLM audit |
| **Sloan (1996), Francis et al. (2005)** | Konvence vyřadit SIC 6000–6999 (finance). | Sample |

---

## 3. Data (`03_data.tex`)

### 3.1 Výběr vzorku (`sec:sample_selection`)
- **CO:** 1000 největších US firem dle market capu → 3 filtry → 535 firem (539 klastrů v některých specifikacích).
- **Filtry:**
  1. SEC „operating" (vyřazují se ETF, shell entities).
  2. **Vyřazen SIC 6000–6999** (finance, real estate) — standardní praxe (Sloan 1996, Francis 2005).
  3. ≥5 let XBRL + ≥90 % return coverage.
- **PROČ 535 a ne víc:** clusterované SE závisí na počtu *klastrů* (Petersen 2009), 539 ≫ prahu. Výpočetní náročnost (parsování 10-K). Malé firmy = vyšší šum (kratší XBRL historie, tagging errors, tenké obchodování).
- **Srovnání:** Campbell (~9 000 firm-years), Loughran (~50 000) — ale tam jde o slovník-validaci, ne joint test.

### 3.2 Proč roční 10-K a ne kvartální 10-Q (`sec:annual_vs_quarterly`)
- **Problém 1 — finanční mezery v Q4:** 10-Q pokrývá jen Q1–Q3, Q4 se musí dopočítat reziduem (annual − Q1−Q2−Q3). Pro provozní CF jsou všechna 3 kvartální čísla dostupná jen v **7 %** firm-years (většina firem reportuje YTD, ne standalone Q).
- **Problém 2 — chybějící risk factors v 10-Q:** SEC Reg. S-K vyžaduje update jen při „material changes" — v pilotu (15 firem, 90 filingů) jen ~50 % 10-Q mělo skutečnou sekci.
- **Závěr:** roční panel je jediná možnost, jak postavit rok-na-rok textové proměnné.

### 3.3 XBRL tag konzistence (`sec:xbrl_tags`) — **vlastní metodologický příspěvek**
- **Problém:** jeden ekonomický koncept = více US-GAAP tagů napříč roky/firmami.
  - Provozní CF: `NetCashProvidedByUsedInOperatingActivities` (total) vs `...ContinuingOperations`.
  - Net income: `NetIncomeLoss`, `ProfitLoss`, `IncomeLossFromContinuingOperations`.
  - Equity: s/bez non-controlling interest.
- **Řešení (2 části):**
  1. **Equivalence groups:** tagy lišící se jen verzí taxonomie (ne ekonomicky) sloučím. Ověřeno na **6 113 firm-years** s oběma tagy současně — mediánové zkreslení ROA jen 0,033 pp.
  2. **Per-firm tag locking:** pro ekonomicky odlišné tagy zamknu firmu na nejčastěji používaný tag; ostatní → missing.
- **Výsledek:** kompletnost 5 kontrol z **78,9 % → 92,7 %** firm-year observací.
- **Argumentace:** bez tohoto kroku bych měl buď mizernou kompletnost, nebo šum z mixu tagů v rámci jedné firmy.

### 3.4 Extrakce textu z 10-K (`sec:text_extraction`)
- **Zdroj:** SEC EDGAR Submissions API. Drop 10-K/A amendments (obsahují jen restated exhibit, ne narativní sekce). Finanční data za amended roky zůstávají (XBRL je zachytí).
- **HTML parser (4 fáze):**
  1. Regex detekce „Item X" + formátování (bold, ALL-CAPS, ≥12 pt). Odmítá hyperlinky a TOC (signatura: hustota dalších Item odkazů v okolí 300 znaků).
  2. Označení potvrzených headerů před flattenováním na plain text.
  3. Vybírá header následovaný nejdelším textem do dalšího Item.
  4. Hranice = další Item header (1B/1C/2 pro 1A; 7A/8 pro 7).
- **Exhibit 13 fallback:** pokud sekce < 200 slov, zkusím externí exhibit (typicky Exhibit 13) s title-only patterns.
- **Post-processing:** odstranění „Table of Contents" headerů, samostatných stránkových čísel, nbsp znaků.

### 3.5 Pokrytí, limitace a LLM audit (`ssec:text_coverage`)
- **Ruční validace:** 50 nejnovějších (50/50 OK) + 100 stratifikovaných (97/100 OK; 3 selhání = PDF-only exhibity nebo chybějící formátovací cue).
- **LLM-assisted audit (klíčový kus):**
  - **Google Gemini 2.5 Flash**, 100 filingů × 2 sekce (1A + 7) = **200 judgementů**.
  - Stratifikováno 20 per 5-leté období (2010–2024).
  - 5 kritérií: identita sekce, start/end hranice, kvalita, úplnost. Škála Pass/Minor/Fail.
  - **Výsledek:** 93,5 % Pass+Minor na všech 5 kritériích současně (n = 187/200; 95 % Wilson CI: 89,2–96,2 %).
  - Selhání koncentrována v 2010–2012 (5/40) vs 2022–2024 (1/40) — odráží postupnou standardizaci SEC HTML.
- **Argumentace u obhajoby:** „LLM ne jako generátor, ale jako *nezávislý klasifikátor* (Gilardi 2023). Není to text-mining v textu — je to audit pipeline."

> **Otázka, kterou ti vedoucí může položit:** Proč ne XBRL `us-gaap:RiskFactorsTextBlock`? — **Odpověď:** Inline XBRL je inconsistently filled před 2018, HTML parser pokrývá celý 2010–2024 window.

---

## 4. Metodologie (`04_methodology.tex`)

### 4.1 Závislá proměnná (`sec:outcome`)
```
ln σ_{i,t+1}^{[h]} = ln( √252 · sd(r_{i,d}) )    pro d ∈ [filing+2d, filing+h d]
```
- **2denní lag:** vyloučení same-day reakce na filing.
- **√252:** annualizace denní směrodatné odchylky.
- **log:** symetrizace pravé-skewed distribuce; koeficienty čteme jako % změnu σ.
- **Headline:** h = 30 dní. Další horizonty (5, 10, 90, 180, 365 d) jako explorativní.

### 4.2 Kontroly (`sec:controls`)
- **X_fin:** Size = ln(Total Assets), Leverage = liab/assets, ROA = NI/assets, Asset Growth = ΔAssets/lagged Assets.
- **Lagged σ** s vlastním koeficientem φ — z předchozího filing-roku, stejné okno h.
- **Winsorizace** na 1./99. percentilu finančních poměrů (text se nevinsorizuje — log a ratio jsou bounded).
- **H3 varianta:** asset growth nahrazen 2 likviditními proxy (current ratio, OCF/assets) — Kravet & Muslu 2013.

### 4.3 H1 (`sec:h1`) — délka Item 1A
- `Words₁ₐ` = počet tokenů v Item 1A; do regrese vchází `ln(Words₁ₐ)`.
- Medián ~8 800 slov, dlouhý pravý ocas.
- **H1 testovaná v jedné regresi s H2** (joint specification, eq. 4.3) — bez toho by se kanály nedaly oddělit (RiskDensity má délku ve jmenovateli).

### 4.4 H2 (`sec:h2`) — RiskDensity a její dekompozice
**První stupeň (composite):**
```
RiskDensity = (n_unc + n_lit) / Words₁ₐ
```
- Vyšší hodnoty = vyšší podíl generického rizikového slovníku → méně firm-specific info.

**Druhý stupeň (decomposition):**
```
UncDensity = n_unc / Words₁ₐ
LitDensity = n_lit / Words₁ₐ
```
- **UncDensity:** hedging („may", „could", „uncertain") — management neví, co se stane.
- **LitDensity:** právní safe-harbour boilerplate („litigation", „indemnify", „breach") — Nelson & Pritchard 2007.
- **Korelace mezi nimi r ≈ −0,05** → separátně identifikovatelné.
- **H2 podpořena**, pokud LitDensity < 0 a UncDensity ≈ 0.

### 4.5 H3 (`sec:h3`) — peer-relativní omission gap
**Konstrukce ve 3 krocích:**
1. **Peer-cell:** firmy ve stejném SIC2 a fiskálním roce (min 5 firem; medián 35). Leave-one-out (focal firma vyloučena z vlastních peerů).
2. **TF-IDF:** každý Item 1A → TF-IDF vektor. Filtry: term v ≥5 a ≤95 % filingů; L2 normalizace.
3. **Omission gap:**
```
Omission₁ₐ = 1 − (peer-mass na termech, které firma zmiňuje / celková peer-mass)
```
- 0 = firma pokrývá vše, co píší peers; 1 = firma ignoruje vše.

**Specifikace (eq. 4.7):** přidává `β_om · Omission₁ₐ` k joint H1+H2 modelu.

- **Dvě FE struktury:**
  - **Cross-section:** SIC2 + rok FE.
  - **Within-firm:** firma + rok FE — využívá jen meziroční změny v omission.
- **Pozor:** korelace `Omission₁ₐ` vs `ln(Words₁ₐ)` = **−0,85** → nutná kontrola délky.
- **PROČ SIC2:** standard v disclosure-similarity literatuře (Brown & Tucker 2011, Dyer 2017).

### 4.6 Estimace a inference (`sec:estimation`)
- **OLS** s absorbovanými FE přes `linearmodels` (Python).
- **Identifikace:** within-industry-year (nebo within-firm-year) variace — vyřazuje crisis shocks (2008, 2020) a perzistentní industry rozdíly.
- **Dvouvrstvové clusterování SE:** firma × rok (Petersen 2009, Cameron 2011) — kvůli (i) překryvu firma-rok oken (within-firm serial corr), (ii) calendar-year shockům (COVID).
- **Sample H1+H2 (h = 30 d):** 5 691 firm-years, 535 firem. **H3 sample:** 4 630–4 636 (menší kvůli ≥5 peerů restrikci).
- **Argumentace u obhajoby:** „Prediktivní čtení (Campbell-style). Koeficienty nejsou kauzální efekty."

---

## 5. Výsledky (`05_results.tex`)

### 5.1 Deskriptivní statistiky (`sec:descriptive`)
- Mean `ln σ ≈ −1,34` → typická annualizovaná σ ≈ 0,26.
- Medián Item 1A ≈ 8 800 slov.
- Korelace UncDensity vs LitDensity ≈ −0,05 (separátně identifikovatelné).
- Korelace `Words₁ₐ` vs RiskDensity = **−0,30** — delší filing má nižší density (délka je dělaná firm-specific textem, ne větším počtem generických slov).

### 5.2 Hlavní výsledky H1+H2 (`sec:joint_results`)
**Tabulka `main_30d`, 5 nested specifikací:**
| Sloupec | Co testuje | Výsledek |
|---|---|---|
| (1) | Finanční baseline | — |
| (2) | Jen délka | β_len > 0 |
| (3) | Jen density | β_dens < 0 |
| (4) | **Joint H1+H2** | β_len = +0,121 (t = +7,64); β_dens = −3,88 (t = −2,68) |
| (5) | **LM dekompozice** | β_lit = −6,16 (t = −3,38, p < 0,001); β_unc = −1,28 (n.s.) |

**Klíčové zjištění:**
- Density koeficient se **mezi (3) a (4) půlí** — polovina toho, co vypadalo jako „density efekt", byla ve skutečnosti volume efekt skrze délkový jmenovatel.
- Negativní specificity efekt **nese výhradně LM-Litigious** (boilerplate), ne LM-Uncertainty.
- **Wald test β_unc = β_lit:** z = +1,91, p = 0,056 (hraniční).
- Adjusted R² roste jen mírně (0,492 → 0,507) — finanční baseline už pojme většinu predikovatelné σ.

> **Argumentace:** „Trh ignoruje právní boilerplate (opakující se 'may be subject to litigation'), ale reaguje na firm-specific obsah zachycený délkou."

**Tabulka `main_365d`:** stejných 5 sloupců na ročním horizontu, znaménka zachována, magnitudy ~poloviční.

### 5.3 H3 — strategické mlčení (`sec:silence_results`)
**Tabulka `silence_main`:**
- **Panel A (industry + rok FE):** β_om ≈ 0 na všech horizontech (|t| < 1,1) — **null**.
- **Panel B (firma + rok FE):** β_om = **+1,345** (t = +2,93, p = 0,003) na 30 d; přežívá na 90 a 180 d.
- **Ekonomická velikost:** +1 within-firm SD (0,054) ≈ **+7,3 % na 30denní σ** — stejný řád jako H1 length efekt.

**Interpretace (proč firm-FE mění výsledek):**
- **Mezi firmami:** velké stabilní rozdíly v šíři risk profilu (IT vs utility) — trvale různé Omission → β ≈ 0.
- **Within-firm:** co se změní rok-od-roku v tom, co firma vynechá relativně k vlastnímu průměru — TAM je informace o budoucí σ.

> **Argumentace:** „Permanentní firm-level disclosure styly absorbují cross-section. Peer-relativní omission gap, který se z konstrukce mění s tím, co píší *jiní*, přežívá firm-FE filtr."

---

## 6. Robustnost (`06_robustness.tex`)

### 6.1 Alternativní horizonty (`sec:robustness_horizons`)
**Tabulka `horizons` (5, 10, 30, 90, 180, 365 d):**
- Všechna 3 textová znaménka zachována na všech horizontech.
- LitDensity signifikantní na 1 % od 10 d (|t| ∈ [2,88; 4,02]).
- UncDensity nikdy nedosáhne signifikance (|t| ≤ 1,1).
- Koeficienty **monotónně klesají s h** — typický signature event-driven signálu (Loughran 2011, Tetlock 2007).

### 6.2 Pre-COVID vs COVID-era (`sec:robustness_subperiods`)
- Pre-COVID (≤2019, n = 3 243): β_len = +0,103 (t = +6,06); β_lit = −5,49 (t = −2,55).
- COVID-era (≥2020, n = 2 448): β_len = +0,172 (t = +8,92); β_lit = −9,88 (t = −4,85).
- UncDensity n.s. v obou.
- **Závěr:** efekt v obou obdobích, COVID-era jen větší magnituda (vyšší overall σ).

### 6.3 Firm fixed effects (`sec:robustness_firmfe`)
**Tabulka `firm_fe`:**
- β_len přežívá: **+0,084 (t = +3,31)** — firmy, které rok-od-roku rostou v Item 1A, mají vyšší následnou σ.
- Obě sub-density absorbovány (β_unc = −1,46, t = −0,55; β_lit = +0,96, t = +0,35).
- **Vysvětlení:** Item 1A se rok-od-roku recykluje (Cazier 2021) → boilerplate má skoro nulovou within-firm variaci → není z čeho identifikovat sub-density.
- **Důsledek:** H2 čteme jako **between-firm regularitu**, ne within-firm kauzální kanál.

### 6.4 Robustnost H3 (`sec:robustness_silence`)
**Tabulka `silence_robust`:**

| Sloupec | Test | Výsledek |
|---|---|---|
| Headline | Within-firm FE, 30 d | +1,345 (t = +2,93) |
| **Placebo (peer-cell shuffle)** | Náhodné permutace SIC2 cell, 5 nezávislých draws | \|t\| ∈ [0,54; 0,90] **«** real t = +2,93 |
| (3) Length-orthogonalized | Omission residual po regresi na ln(Words) | t = +1,49 (hraniční) |
| (5) + 2 length kontroly | Přidá MD&A word count + (ln Words)² | t = +3,18 (silnější) |
| (4) Large-cell ≥20 peerů | n = 3 248, lepší peer aggregate | β = **+2,52** (t = +3,09) — **silnější signál** |
| (6) Prior-year Omission | Adresuje filing-window overlap | β = +0,485 (t = +1,97) |
| (2) Decile rank | Místo úrovně decil v cellu | t = +1,31 (slabé) |

- **Placebo je nejdůležitější:** kdyby šlo o length artefakt, placebo by zrcadlilo real. Místo toho real |t| **3× převyšuje** maximum placebo. → mechanismus je *peer-specifický*.
- **Decile-rank slabost** = nejdůležitější kvalifikace: efekt je v *ocasech* distribuce (konspicuózní omissions), ne lineární.

### 6.5 Co nefungovalo (`sec:robustness_failed`)
7 alternativních specificity kandidátů testovaných a zamítnutých:
1. Year-on-year TF-IDF similarity Item 1A (Brown 2011).
2. MD&A LM-Negativity.
3. Δ ln(Words₁ₐ) (Lyle 2023).
4–7. **FinBERT-tone** 4 funkční formy (P_neg, P_pos − P_neg, P_neutral, 1 − P_neutral).

Žádný nepřidal signifikantní explanatory power (|t| < 1,5). LM dekompozice uspěla, protože identifikuje **strukturální složení H2 efektu**, ne další firm-specific signál konkurující H1.

> **Argumentace:** „Negativní výsledky jsou poctivý report. FinBERT zní moderně, ale na tomto panelu je dominován délkovým signálem."

---

## 7. Závěr (`zaver.tex`)

**3 hlavní zjištění:**
1. **Objem ≠ specifičnost.** Oba kanály jsou současně identifikovatelné v jedné regresi. Délka = riziko; density bez kontroly délky = částečně artefakt.
2. **Boilerplate, ne hedging.** Negativní specificity efekt drží LM-Litigious slovník — právní safe-harbour wording, který trh správně diskontuje.
3. **Mlčení vůči peerům informuje.** Within-firm změny v peer-relativním Omission predikují σ. Cross-section to nezachytí, protože firm-level disclosure style je perzistentní.

**Limity (otevřeně přiznat):**
- Prediktivní, ne kauzální čtení.
- Vzorek velkých US firem — externí validita k malým a non-US omezená.
- H3 závisí na volbě peer-benchmarku (SIC2) a v decilech slábne.
- LM slovník je dictionary-based — nezachytí kontextové významy.

**Přínos:**
- Replikovatelná pipeline (kód v `scripts/`).
- Tag-locking metodologie pro XBRL (vlastní příspěvek).
- LLM-assisted audit jako šablona validace text-mining pipeline.

---

## 8. Otázky, na které musíš mít připravenou odpověď

1. **„Proč prediktivní a ne kauzální?"** — Nemáme exogenní variaci v disclosure. Identifikace probíhá ze rezidualní variace po finančních kontrolách a FE.
2. **„Proč LM, ne moderní transformer (FinBERT, RoBERTa)?"** — Zkoušeno (sekce 6.5), nepřidává explanatory power. LM je transparentní a replikovatelný.
3. **„Není H1 jen proxy pro řadu rizikových faktorů?"** — Ano, do jisté míry. Proto se H2 dekompozice ptá *na složení* textu při kontrole délky.
4. **„Proč 30 dní?"** — Pre-specified headline. Sekce 6.1 ukazuje signál na všech horizontech.
5. **„Není Omission gap jen inverz délky (r = −0,85)?"** — Adresováno: (a) kontrola `ln(Words)` v regresi, (b) length-orthogonalized varianta (Col 3), (c) placebo na peer-cell shuffle.
6. **„Proč SIC2 a ne user-supplied peers (10-K filings)?"** — SIC2 = standard (Brown 2011, Dyer 2017); user-peers nejsou k dispozici pro celý 2010–2024 window.
7. **„Není 535 firem málo?"** — Petersen 2009: clusterování závisí na počtu *klastrů* (539), ne pozorování. Campbell (~9k firm-years) je podobný řád.
8. **„Co Gemini? Není LLM unreliable?"** — Použito jen jako *klasifikátor* na pre-extracted text, ne generátor. 93,5 % shoda s ruční validací, podpořeno Gilardi 2023.
9. **„Proč nepoužít XBRL `RiskFactorsTextBlock`?"** — Před 2018 inconsistently filled. HTML parser pokrývá celý window.
10. **„Wald test (z = 1,91, p = 0,056) je hraniční — je dekompozice statisticky podpořená?"** — Ano, qualitativně silně (LitDensity p < 0,001, UncDensity p ≈ 0,31); rozdíl velikosti hraniční. Robustní na všech horizontech a v obou sub-obdobích.

---

## 9. Číselný cheat-sheet

| Metrika | Hodnota |
|---|---|
| Firem | 535 (539 klastrů) |
| Období | 2010–2024 (15 let) |
| Firm-years H1+H2 (h = 30 d) | 5 691 |
| Firm-years H3 | 4 630–4 636 |
| Headline horizont | 30 dní (post-filing, +2 d lag) |
| **β_len** | +0,121 (t = +7,64) |
| **β_dens** | −3,88 (t = −2,68) |
| **β_lit** | −6,16 (t = −3,38) |
| **β_unc** | −1,28 (t = −1,01) |
| **β_om (firm-FE)** | +1,345 (t = +2,93) |
| β_len (firm-FE) | +0,084 (t = +3,31) |
| β_om (large-cell ≥20 peerů) | +2,52 (t = +3,09) |
| Placebo |t| (5 draws) | 0,54–0,90 |
| Wald z (β_unc = β_lit) | 1,91 (p = 0,056) |
| Tag-locking kompletnost | 78,9 % → 92,7 % |
| LLM audit pass+minor | 93,5 % (Wilson CI 89,2–96,2 %) |
| Medián Item 1A | ~8 800 slov |
| Medián peer-cell | 35 firem (P25 = 14) |
| Korelace Omission vs ln(Words) | −0,85 |
| Korelace UncDensity vs LitDensity | −0,05 |
| Adj. R² (col 4 → 5, h = 30 d) | 0,492 → 0,507 |

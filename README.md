# PAC-SIMULATOR

\
SIMULATORE PAC – DOCUMENTAZIONE & HOW-TO (README.txt)
=====================================================

Autore: Luca
\
App: Streamlit – “Simulatore di PAC (Piano di Accumulo)”

## Indice

1. Panoramica

2. Requisiti & Installazione

3. Avvio rapido

4. Logica finanziaria del simulatore

5. Modalità di rendimento (Manuale vs ETF)

6. Monte Carlo – come funziona

7. Interfaccia & flusso d’uso (HOW‑TO utente)

8. Output: grafici e tabelle

9. Troubleshooting (errori comuni)

10. Note per repository Git (struttura consigliata)

11. Licenza & avvertenze

12. Panoramica

---

1. Panoramica


Web app minimal/modern in Streamlit per simulare un PAC.
Funzioni chiave:

* Contributo mensile con incremento annuo fino a un tetto massimo.
* Costi: TER (mensile), bollo (annuale), tasse (solo alla fine su plusvalenza).
* Rendimento: manuale o da ETF (media 5Y ponderata sui pesi e TER).
* Monte Carlo opzionale con banda di confidenza, rendimenti annui netti log‑normali e volatilità di portafoglio con covarianza.

---

2. Requisiti & Installazione


Python ≥ 3.9

Librerie:

* streamlit
* pandas
* numpy
* yfinance
* plotly

Installazione (consigliata in virtual env):

```
pip install streamlit pandas numpy yfinance plotly
```

---
3. Avvio rapido


- Posizionati nella cartella del progetto.-
-  Avvia:

```
streamlit run main.py
```

- Il browser si aprirà su [http://localhost:8501](http://localhost:8501)

---
4. Logica finanziaria del simulatore


* Contributi: versamento mensile; ogni 12 mesi cresce di “Incremento annuo %” fino a “Tetto max mensile €”.
* Rendimento: si applica mensilmente a capitali e contributi (tasso mensile derivato dal tasso annuo).
* TER: applicato come fee **mensile** (TER annuo / 12) sul valore del portafoglio; riduce la base investita in corso d’opera.
* Bollo: applicato **a fine anno** come % sul valore di fine anno.
* Tasse: **solo alla fine** della simulazione, su plusvalenza realizzata:
  Plusvalenza tassabile = (Valore netto *pre-tasse* finale) − (Investito cumulato).
  Imposta finale = Plusvalenza tassabile × 26%.

Campi principali:

* **Valore lordo cumulato**: evoluzione senza costi.
* **Valore netto (pre-tasse)**: dopo TER e bollo, prima dell’imposta finale.
* **Valore netto finale**: netto pre‑tasse − imposta finale.
* Il gap tra lordo e netto pre‑tasse > (TER + bollo) perché include anche il **drag da compounding** (rendimenti persi per costi avvenuti prima).

---

5. Modalità di rendimento (Manuale vs ETF)


Manuale:

* Inserisci rendimento annuo **lordo** atteso (%) e TER medio del portafoglio.
* L’app mostra anche il rendimento **netto** implicito (= lordo − TER).

ETF (media 5 anni):

* Inserisci uno o più ticker con **suffisso borsa** (es. VNGA40.MI, V40A.DE).
* L’app risolve i ticker provando suffissi comuni e alias; blocca la simulazione se non trova il simbolo.
* Imposta i **pesi**: la somma deve essere 100%.
* Scegli la metrica storica per ogni ETF: **Media annua (aritmetica)** o **CAGR (geometrica)**; l’app calcola la **media ponderata** del portafoglio.
* TER portafoglio = **media ponderata** dei TER per ETF.
* Rendimento lordo usato = media netta ponderata + TER ponderato.

---
6. Monte Carlo – come funziona


Obiettivo: mostrare un **range plausibile** (banda di confidenza) intorno all’andamento medio.

Meccanica:

* Ogni **anno** si estrae un rendimento **netto** da una **lognormale** parametrizzata su media (μ) e volatilità (σ) storiche del portafoglio.
* I rendimenti estremi sono **clippati** (es. tra −40% e +40%) per restare realistici.
* Si ricostruisce il rendimento **lordo** aggiungendo il TER medio e si converte a **mensile** in forma logaritmica (stabile numericamente).
* Si applicano i versamenti, il tasso mensile, i **TER mensili** e il **bollo** a fine anno. Si memorizza il **netto pre‑tasse** anno per anno.
* Dopo N simulazioni (es. 1000), per ogni anno si calcolano i percentili (es. 10°–90%) → banda; la curva tratteggiata è la **mediana**.

Parametri:

* μ annuo netto (Manuale: lordo − TER; ETF: media storica ponderata).
* σ annuo **di portafoglio**: calcolato da **covarianza** dei rendimenti (√(wᵀ Σ w)) su dati giornalieri (annualizzati).
* Banda di confidenza: 5–95% o 10–90% (consigliata 10–90 per stabilità visiva).

---

7. Interfaccia & flusso d’uso (HOW‑TO utente)


  - **Scegli la modalità**: Manuale oppure ETF (media 5 anni).
  - **Imposta durata** (anni) e contributi (mensile iniziale, incremento annuo %, tetto massimo).
  - **Costi**: imposta **aliquota capital gain** (tipicamente 26%) e **bollo** (tipicamente 0,2%).
  - Se **Manuale**: inserisci rendimento **lordo** e **TER**; opzionale: **volatilità annua** per Monte Carlo.
  - Se **ETF**: compila tabella ticker/pesi/TER; verifica che i pesi sommino a 100%; seleziona metrica (CAGR o media annua).
  - (Opzionale) **Monte Carlo**: spunta il checkbox, scegli N simulazioni, banda e seed.
  - Clicca **Esegui simulazione**.
  - Leggi i risultati:
  
     * **Guadagni**: totale investito, lordo, guadagno netto.
     * **Costi**: costi totali, TER totali, tasse finali, bollo totale.
     * **Grafico**: investito, lordo, netto; se MC attivo, **banda di confidenza** + mediana MC.
     * **Tabella annuale**: investito anno, cumulati, netto pre‑tasse.
     * **Nota fiscale** con i numeri della simulazione (plusvalenza tassabile, imposta e netto finale).
  - (Opzionale) Scarica il CSV del dettaglio annuale.

---

8) Output: grafici e tabelle


* Grafico linee (Plotly): Investito cumulato, Valore lordo, Valore netto. MC: area confidenza + mediana.
* Tabella annuale: Anno, Investito anno, Investito cumulato, Valore lordo cumulato, Valore netto (pre‑tasse), TER cumulato, Bollo cumulato.
* KPI responsive: “Guadagni” e “Costi” separati.

---

9. Troubleshooting (errori comuni)


* **Ticker non risolto**: usa suffisso borsa (es. *.MI*, *.DE*). Per LifeStrategy: VNGA40.MI / V40A.DE.
* **Pesi ≠ 100%**: correggi la somma; l’app blocca l’esecuzione.
* **yfinance non disponibile**: `pip install yfinance`.
* **Valori complessi / errori Monte Carlo**: protezione già inclusa (logaritmica + clip). Se σ è 0, la banda sarà piatta.
* **Banda sotto il netto con molti ETF**: ora σ portafoglio usa **covarianza** (√(wᵀ Σ w)).
* **Stampe inattese (True/False)**: evitare espressioni “nude” in Streamlit; assegnare a variabili o rimuovere debug.

---

10. Note per repository Git (struttura consigliata)


```
.
├─ main.py
├─ requirements.txt           # streamlit, pandas, numpy, yfinance, plotly
├─ README.md                  # versione Markdown di questo file
├─ LICENSE                    # MIT o altra
├─ .gitignore                 # venv/, __pycache__/, .streamlit/, .DS_Store
├─ docs/
│   └─ screenshots/           # screenshot UI
```


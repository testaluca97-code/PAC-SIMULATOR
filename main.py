import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Optional: yfinance/plotly for ETF mode + charts
try:
    import yfinance as yf  # for historical ETF data
    YF_AVAILABLE = True
except Exception:
    YF_AVAILABLE = False

try:
    import plotly.express as px  # nicer interactive charts
    PLOTLY_AVAILABLE = True
except Exception:
    PLOTLY_AVAILABLE = False

# -----------------------------
# Page & Styles
# -----------------------------
st.set_page_config(
    page_title="Simulatore PAC",
    page_icon="💶",
    layout="wide",
)

CUSTOM_CSS = """
<style>
:root{
  --bg:#ffffff; 
  --card:#f8fafc; 
  --card-border:rgba(0,0,0,0.06);
  --muted:#475569; 
  --primary:#2563eb; 
  --text:#0f172a; 
  --good:#047857; 
  --bad:#b91c1c; 
}

.block-container {padding-top: 1.5rem;}
html, body, [class^="css"] {background: var(--bg) !important;}

h1,h2,h3,h4,h5,h6, p, span, div, label { color: var(--text) !important; }

/* Responsive sections */
.section-grid{ display:grid; grid-template-columns: repeat(auto-fit, minmax(200px,1fr)); gap:14px; margin: 10px 0 16px; }
.card{ background: var(--card);
       border:1px solid var(--card-border); border-radius:16px; padding:16px; box-shadow: 0 6px 12px rgba(0,0,0,0.06);} 
.card .label{ font-size: 0.83rem; color: var(--muted) !important; }
.card .value{ font-weight:700; font-size:1.3rem; margin-top:4px; }
.card.good .value{ color: var(--good) !important; }
.card.bad .value{ color: var(--bad) !important; }
.card.neutral .value{ color: var(--primary) !important; }

/* Data editor tweaks */
[data-testid="stDataFrame"] table{ border-radius: 12px; overflow:hidden; }
.accent{ color:#2563eb !important; font-weight:700; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------

def eur(x: float) -> str:
    try:
        return f"€{x:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    except Exception:
        return f"€{x:.2f}"

@st.cache_data(show_spinner=False)
def fetch_5y_metrics(ticker: str):
    """Risoluzione automatica del ticker su Yahoo Finance + metriche 5Y.
    Ritorna: avg_arith (media annua), cagr (~5Y), resolved ticker, tentativi.
    """
    if not YF_AVAILABLE:
        raise RuntimeError("yfinance non disponibile. Installa 'yfinance'.")

    raw = ticker.strip().upper()

    # Alias utili per Vanguard LifeStrategy su diverse borse
    alias = {
        "VNGA20": ["VNGA20.MI", "V20A.DE"],
        "VNGA40": ["VNGA40.MI", "V40A.DE"],
        "VNGA60": ["VNGA60.MI", "V60A.DE"],
        "VNGA80": ["VNGA80.MI", "V80A.DE"],
    }

    suffixes = ["", ".MI", ".IM", ".DE", ".L", ".AS", ".PA", ".SW", ".IR", ".TO", ".SA", ".HK", ".AX", ".VI", ".ST", ".LS", ".MC", ".OL", ".CO", ".HE", ".IS"]

    candidates = [raw]
    candidates += alias.get(raw, [])
    if "." not in raw:
        candidates += [raw + s for s in suffixes if s]

    tried, resolved, df = [], None, None
    for sym in dict.fromkeys(candidates):
        try:
            dft = yf.download(sym, period="6y", interval="1d", auto_adjust=True, progress=False)
            tried.append(sym)
            if dft is not None and not dft.empty:
                resolved, df = sym, dft
                break
        except Exception:
            tried.append(sym)
            continue

    if resolved is None or df is None or df.empty:
        raise RuntimeError(
            f"Ticker non risolto: {raw}. Usa il suffisso borsa (es. VNGA40.MI). Tentativi: {', '.join(tried)}"
        )

    # Media annua (ultimi 5 anni solari completi)
    annual = df["Close"].resample("YE").last().pct_change().dropna()
    last5 = annual.tail(5)
    avg_arith = float(last5.mean()) if len(last5) > 0 else np.nan

    # CAGR su ~5 anni
    start_idx = df.index.max() - pd.DateOffset(years=5)
    df5 = df.loc[df.index >= start_idx]
    if df5.empty:
        df5 = df
    start_val = float(df5["Close"].iloc[0])
    end_val   = float(df5["Close"].iloc[-1])
    years = max(1e-9, (df5.index[-1] - df5.index[0]).days / 365.25)
    cagr = (end_val / start_val) ** (1/years) - 1

    return {
        "ticker": raw,
        "resolved": resolved,
        "tried": tried,
        "avg_arith": avg_arith,
        "cagr": cagr,
        "years_measured": years,
    }

# -----------------------------
# Simulatore PAC
# -----------------------------


def simulate_pac(
    years:int,
    start_monthly:float,
    annual_increase:float,
    cap_monthly:float,
    gross_return_yr:float,   # rendimento LORDO annuo usato in simulazione
    ter_weighted:float,      # TER annuo medio ponderato (come %)
    tax_rate:float,          # aliquota capital gain (26%)
    bollo_rate:float,        # imposta di bollo annua (% del valore a fine anno)
    invest_year = 0.0
):
    months = years * 12
    r_m_gross = (1 + gross_return_yr) ** (1/12) - 1
    ter_m = ter_weighted / 12.0

    m_dates, invested_cum_m, gross_m, net_m = [], [], [], []
    year_rows = []

    value_gross = 0.0
    value_after_ter = 0.0
    invested_cum = 0.0

    ter_paid_cum = 0.0
    bollo_paid_cum = 0.0

    today = pd.Timestamp.today().normalize()
    start_date = today.replace(month=1, day=1)
    monthly_contrib = start_monthly

    for m in range(months):
        dt = start_date + pd.DateOffset(months=m)
        m_dates.append(dt)

        # Incremento annuo con tetto
        if m>0 and m % 12 == 0:
            monthly_contrib = min(cap_monthly, monthly_contrib * (1 + annual_increase))

        # Versamento
        value_gross += monthly_contrib
        value_after_ter += monthly_contrib
        invested_cum += monthly_contrib
        invest_year += monthly_contrib

        # Crescita lorda
        value_gross *= (1 + r_m_gross)
        value_after_ter *= (1 + r_m_gross)

        # TER mensile (fee esplicita)
        ter_fee = value_after_ter * ter_m
        value_after_ter -= ter_fee
        ter_paid_cum += ter_fee

        # Fine anno: applica SOLO bollo
        if (m + 1) % 12 == 0:
            bollo_fee = value_after_ter * bollo_rate
            value_after_ter -= bollo_fee
            bollo_paid_cum += bollo_fee

            year_rows.append({
                "Anno": start_date.year + (m+1)//12 - 1,
                "Investito anno": invest_year,
                "Investito cumulato": invested_cum,
                "Valore lordo cumulato": value_gross,
                "Valore netto (pre-tasse)": value_after_ter,
                "TER cumulato": ter_paid_cum,
                "Bollo cumulato": bollo_paid_cum,
                "Tasse (anno)": 0.0,  # tassazione solo alla fine
            })
            # reset per l'anno successivo
            invest_year = 0.0

        invested_cum_m.append(invested_cum)
        gross_m.append(value_gross)
        net_m.append(value_after_ter)

    # --- TASSA FINALE ALLA VENDITA ---
    gain_tot = max(0.0, value_after_ter - invested_cum)
    taxes_tot = gain_tot * tax_rate
    value_after_tax = value_after_ter - taxes_tot

    # Aggiorna ultimo punto mensile con le tasse finali
    if len(net_m) > 0:
        net_m[-1] = value_after_tax

    # Aggiorna ultima riga annuale con tassa finale
    if len(year_rows) > 0:
        last = year_rows[-1].copy()
        last["Tasse (anno)"] = taxes_tot
        last["Valore netto (pre-tasse)"] = value_after_ter
        last["Valore netto (fine anno)"] = value_after_tax
        year_rows[-1] = last

    totals = {
        "investito_tot": invested_cum,
        "lordo_tot": value_gross,
        "ter_tot": ter_paid_cum,
        "tasse_tot": taxes_tot,
        "bollo_tot": bollo_paid_cum,
        "netto_tot": value_after_tax,
    }
    totals["costi_tot"] = totals["ter_tot"] + totals["tasse_tot"] + totals["bollo_tot"]
    totals["guadagno_netto"] = totals["netto_tot"] - totals["investito_tot"]

    monthly_df = pd.DataFrame({
        "Data": m_dates,
        "Investito cumulato": invested_cum_m,
        "Valore lordo": gross_m,
        "Valore netto": net_m,
    })

    yearly_df = pd.DataFrame(year_rows)
    return monthly_df, yearly_df, totals

# -----------------------------
# Monte Carlo helpers (aggiunte)
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_5y_annual_returns_series(ticker: str):
    if not YF_AVAILABLE:
        return pd.Series(dtype=float)
    try:
        m = fetch_5y_metrics(ticker)
        resolved = m["resolved"]
    except Exception:
        return pd.Series(dtype=float)
    df = yf.download(resolved, period="6y", interval="1d", auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.Series(dtype=float)
    annual = df["Close"].resample("YE").last().pct_change().dropna()
    return annual.tail(5)



def _draw_net_return_lognorm(rng, mu_ann, sigma_ann, clip_floor=-0.4, clip_cap=0.4):
    """
    Estrae un rendimento NETTO annuo da lognormale, poi clippa tra clip_floor e clip_cap.
    mu_ann, sigma_ann sono media e dev. std ARITMETICHE del rendimento netto (R).
    Converte a parametri lognormali su Y=1+R:
        s2 = ln(1 + sigma^2 / (1+mu)^2)
        m  = ln(1+mu) - 0.5*s2
    Poi: Y = exp(N(m,s))  => R = Y - 1
    """
    # Protezione su limiti ragionevoli
    mu_ann = float(mu_ann)
    sigma_ann = float(max(sigma_ann, 1e-9))
    one_plus_mu = 1.0 + mu_ann
    s2 = np.log(1.0 + (sigma_ann**2) / (one_plus_mu**2))
    m  = np.log(one_plus_mu) - 0.5 * s2
    s  = np.sqrt(s2)
    Y  = np.exp(rng.normal(m, s))      # Y = 1+R
    R  = Y - 1.0                       # netto annuo
    # Clipping per evitare code troppo estreme
    return float(np.clip(R, clip_floor, clip_cap))


def simulate_pac_stochastic(
    years:int,
    start_monthly:float,
    annual_increase:float,
    cap_monthly:float,
    mu_ann:float,       # media NETTA annua
    sigma_ann:float,    # deviazione std annua
    ter_weighted:float,
    tax_rate:float,
    bollo_rate:float,
    runs:int=1000,
    seed:int=42,
):
    rng = np.random.default_rng(seed)
    months = years * 12
    paths_pre_tax_year = np.zeros((runs, years))

    for r in range(runs):
        value_gross = 0.0
        value_after_ter = 0.0
        invested_cum = 0.0
        monthly_contrib = start_monthly

        for m in range(months):
            if m > 0 and m % 12 == 0:
                monthly_contrib = min(cap_monthly, monthly_contrib * (1 + annual_increase))
            if m % 12 == 0:
                # Estrazione da LOGNORMALE + clipping (più realistica)
                # NB: ret_y è NETTO annuo (dopo TER). Ricostruiamo il LORDO aggiungendo TER.
                # Puoi cambiare i limiti se vuoi bande più strette/larghe.
                ret_y = _draw_net_return_lognorm(rng, mu_ann, sigma_ann, clip_floor=-0.40, clip_cap=0.40)
                ann_gross = ret_y + ter_weighted
                # versione logaritmica stabile per il passaggio a mensile
                r_m_gross = np.expm1(np.log1p(ann_gross) / 12.0)
                ter_m = ter_weighted / 12.0


            # versamento
            value_gross     += monthly_contrib
            value_after_ter += monthly_contrib
            invested_cum    += monthly_contrib

            # crescita lorda
            value_gross     *= (1 + r_m_gross)
            value_after_ter *= (1 + r_m_gross)

            # TER mensile
            ter_fee = value_after_ter * ter_m
            value_after_ter -= ter_fee

            # fine anno: bollo + snapshot pre-tasse
            if (m + 1) % 12 == 0:
                bollo_fee = value_after_ter * bollo_rate
                value_after_ter -= bollo_fee
                paths_pre_tax_year[r, (m + 1)//12 - 1] = value_after_ter

    return paths_pre_tax_year




# -----------------------------
# UI
# -----------------------------
st.title("💶 Simulatore di PAC (Piano di Accumulo)")
st.caption("© Luca")

with st.sidebar:
    st.subheader("📥 Parametri di simulazione")
    mode = st.radio(
        "Sorgente rendimento",
        ["Manuale", "ETF (media 5 anni)"]
    )

    anni = st.slider("Durata (anni)", 1, 40, 15)

    st.markdown("---")
    st.markdown("**Contributi**")
    start_monthly = st.number_input("Contributo mensile iniziale (€)", min_value=0.0, value=200.0, step=50.0)
    annual_up = st.number_input("Incremento annuo (%)", min_value=0.0, value=10.0, step=1.0) / 100.0
    cap_monthly = st.number_input("Tetto massimo mensile (€)", min_value=0.0, value=500.0, step=50.0)

    st.markdown("---")
    st.markdown("**Costi & Fisco (Italia)**")
    tax_rate = st.number_input("Aliquota capital gain (%)", min_value=0.0, max_value=100.0, value=26.0, step=0.5) / 100.0
    bollo_rate = st.number_input("Imposta di bollo annua (%)", min_value=0.0, max_value=2.0, value=0.2, step=0.05) / 100.0

    st.markdown("---")

    if mode == "Manuale":
        st.markdown("**Rendimento atteso**")
        gross_return_yr = st.number_input("Rendimento annuo lordo atteso (%)", min_value=-50.0, max_value=50.0, value=6.0, step=0.5) / 100.0
        ter_weighted = st.number_input("TER medio del portafoglio (%)", min_value=0.0, max_value=3.0, value=0.2, step=0.05) / 100.0
        # Volatilità per Monte Carlo (manuale)
        vol_ann = st.number_input("Deviazione standard annua attesa (%)", min_value=0.0, max_value=60.0, value=15.0, step=0.5) / 100.0
        etf_df = None
        returns_info = None
        metric_name = "Input manuale"
        # Parametri Monte Carlo (manuale)
        st.session_state["mu_ann_net"] = float(gross_return_yr - ter_weighted)  # NETTO
        st.session_state["sigma_ann"]  = float(vol_ann)

    else:
        st.markdown("**ETF & pesi**")
        if not YF_AVAILABLE:
            st.info("Per il calcolo da ETF serve il pacchetto `yfinance`.")
        default_df = pd.DataFrame({
            "Ticker": ["VWCE.MI"],
            "Peso %": [100.0],
            "TER %": [0.22],
        })
        etf_df = st.data_editor(
            default_df,
            num_rows="dynamic",
            width='stretch',
            key="etf_editor",
        )
        # st.caption("Suggerimento: indica il **ticker con suffisso della borsa** — es. *VNGA40.MI* (Borsa Italiana), *V40A.DE* (Xetra).")
        method = st.selectbox("Metodo media 5 anni", ["CAGR (geometrica)", "Media annua (aritmetica)"])

        # Fetch & validate metrics
        returns_info = []
        weighted_ter = 0.0
        weighted_ret_hist = 0.0
        weights_ok = False
        unresolved = []

        if etf_df is not None and len(etf_df) > 0:
            tmp = etf_df.copy()
            tmp = tmp.dropna(subset=["Ticker"]).reset_index(drop=True)
            if len(tmp) > 0:
                # Pesi
                tmp["Peso %"] = np.clip(tmp["Peso %"].fillna(0).astype(float), 0, None)
                weight_sum = float(tmp["Peso %"].sum())
                st.caption(f"Somma dei pesi: **{weight_sum:.2f}%** (deve essere 100%)")
                weights_ok = abs(weight_sum - 100.0) < 0.01
                if not weights_ok:
                    st.warning("La somma delle percentuali deve essere **100%**.")
                # TER ponderato
                ter_vals = np.clip(tmp["TER %"].fillna(0).astype(float).values, 0, None) / 100.0
                w = (tmp["Peso %"].values / 100.0) if weights_ok else None
                if weights_ok:
                    weighted_ter = float((w * ter_vals).sum())

                # Storici & risoluzione ticker
                for _, r in tmp.iterrows():
                    tkr = str(r["Ticker"]).strip().upper()
                    try:
                        m = fetch_5y_metrics(tkr) if YF_AVAILABLE else {"ticker": tkr, "resolved": None, "avg_arith": np.nan, "cagr": np.nan}
                        returns_info.append(m)
                        if not m.get("resolved"):
                            unresolved.append(tkr)
                    except Exception as e:
                        returns_info.append({"ticker": tkr, "resolved": None, "error": str(e), "avg_arith": np.nan, "cagr": np.nan})
                        unresolved.append(tkr)

                # Metrica scelta
                if len(returns_info) > 0 and weights_ok:
                    metric_key = "cagr" if method.startswith("CAGR") else "avg_arith"
                    metric_name = "CAGR" if metric_key == "cagr" else "Media annua (aritmetica)"
                    hist_vec = np.array([ri.get(metric_key, np.nan) for ri in returns_info])
                    hist_vec = np.nan_to_num(hist_vec, nan=0.0)
                    weighted_ret_hist = float((w * hist_vec).sum())
                    # Stima sigma annua portafoglio (approx: media ponderata degli std per-ETF, ignora covarianza)
                    # Stima sigma annua portafoglio con covarianza
                    try:
                        tickers_resolved = []
                        for _, r2 in tmp.iterrows():
                            tkr = str(r2["Ticker"]).strip().upper()
                            m = fetch_5y_metrics(tkr) if YF_AVAILABLE else {"resolved": tkr}
                            if m.get("resolved"):
                                tickers_resolved.append(m["resolved"])

                        if tickers_resolved:
                            # scarica prezzi giornalieri ultimi 5 anni
                            data = yf.download(tickers_resolved, period="5y", interval="1d", auto_adjust=True, progress=False)["Close"]
                            # calcola rendimenti log giornalieri
                            log_rets = np.log(data / data.shift(1)).dropna()
                            # annualizza (dev std * sqrt(252))
                            cov_matrix = log_rets.cov() * 252
                            # calcola σ portafoglio = sqrt(w^T Σ w)
                            sigma_ann = float(np.sqrt(w @ cov_matrix.values @ w.T))
                        else:
                            sigma_ann = 0.0
                    except Exception as e:
                        sigma_ann = 0.0

                    # Parametri Monte Carlo (ETF): μ = media storica NETTA ponderata; σ = stima sopra
                    st.session_state["mu_ann_net"] = float(weighted_ret_hist)
                    st.session_state["sigma_ann"]  = float(sigma_ann)


        # Persist checks
        st.session_state["weights_ok"] = weights_ok
        st.session_state["unresolved"] = unresolved
        st.session_state["returns_info"] = returns_info
        st.session_state["metric_name"] = metric_name if 'metric_name' in locals() else "Media 5Y"

        # Rendimento lordo = media storica (netta) + TER ponderato
        gross_return_yr = weighted_ret_hist + weighted_ter
        ter_weighted = weighted_ter

    st.markdown("---")
    # Controlli Monte Carlo
    mc_enabled = st.checkbox("Attiva simulazione Monte Carlo (banda confidenza)", value=False)
    mc_runs    = st.number_input("N. simulazioni", min_value=200, max_value=5000, value=1000, step=100, disabled=not mc_enabled)
    mc_band    = st.selectbox("Banda confidenza", ["5–95%", "10–90%"], index=1, disabled=not mc_enabled)
    mc_seed    = st.number_input("Seed casuale", min_value=0, value=42, step=1, disabled=not mc_enabled)

    run = st.button("🚀 Esegui simulazione", width='stretch', type="primary")

# -----------------------------
# Run simulation
# -----------------------------
if run:
    if mode == "ETF (media 5 anni)":
        if etf_df is None or len(etf_df.dropna(subset=["Ticker"])) == 0:
            st.error("Inserisci almeno un ETF.")
            st.stop()
        if not st.session_state.get("weights_ok", False):
            st.error("La somma dei pesi deve essere **100%**. Correggi i valori.")
            st.stop()
        unresolved = st.session_state.get("unresolved", [])
        if unresolved:
            st.error("Ticker non risolti: " + ", ".join(unresolved) + ". Esempio: 'VNGA40.MI' (Borsa Italiana) o 'V40A.DE' (Xetra).")
            st.stop()

    monthly_df, yearly_df, totals = simulate_pac(
        years=anni,
        start_monthly=start_monthly,
        annual_increase=annual_up,
        cap_monthly=cap_monthly,
        gross_return_yr=gross_return_yr,
        ter_weighted=ter_weighted,
        tax_rate=tax_rate,
        bollo_rate=bollo_rate,
    )

    # -----------------------------
    # KPI Sections (responsive)
    # -----------------------------
    st.subheader("📊 Risultati")

    # Sezione Guadagni
    gains_html = f"""
    <div class='section-grid'>
      <div class='card neutral'><div class='label'>Totale investito</div><div class='value'>{eur(totals['investito_tot'])}</div></div>
      <div class='card neutral'><div class='label'>Totale lordo (pre-costi)</div><div class='value'>{eur(totals['lordo_tot'])}</div></div>
      <div class='card neutral'><div class='label'>Totale netto</div><div class='value'>{eur(totals['netto_tot'])}</div></div>
      <div class='card good'><div class='label'>Guadagno netto</div><div class='value'>{eur(totals['guadagno_netto'])}</div></div>
    </div>
    """
    net_return = (gross_return_yr - ter_weighted) * 100
    st.markdown(
        f"#### 💰 Guadagni - (rendimento annuo netto &nbsp;&nbsp;<span class='accent'>{net_return:.2f}%</span>)",
        unsafe_allow_html=True
    )
    st.markdown(gains_html, unsafe_allow_html=True)

    # Sezione Costi
    costs_html = f"""
    <div class='section-grid'>
      <div class='card bad'><div class='label'>Costi totali</div><div class='value'>{eur(totals['costi_tot'])}</div></div>
      <div class='card bad'><div class='label'>TER totale</div><div class='value'>{eur(totals['ter_tot'])}</div></div>
      <div class='card bad'><div class='label'>Tasse totali (finali)</div><div class='value'>{eur(totals['tasse_tot'])}</div></div>
      <div class='card bad'><div class='label'>Bollo totale</div><div class='value'>{eur(totals['bollo_tot'])}</div></div>
    </div>
    """
    st.markdown("#### 💸 Costi")
    st.markdown(costs_html, unsafe_allow_html=True)
    # Nota fiscale con numeri della simulazione
    try:
        pre_tax_final = float(yearly_df["Valore netto (pre-tasse)"].iloc[-1]) if not yearly_df.empty else None
    except Exception:
        pre_tax_final = None
    invested_tot = totals.get("investito_tot", 0.0)
    taxes_tot = totals.get("tasse_tot", 0.0)
    net_final = totals.get("netto_tot", 0.0)
    if pre_tax_final is not None:
        taxable_gain = max(0.0, pre_tax_final - invested_tot)
        st.caption(
        f"""
        ℹ️ **Nota fiscale**
        - Le **tasse sul capital gain (26%)** si applicano **solo alla fine** sulla **plusvalenza realizzata**.
        - **Plusvalenza tassabile** = Valore netto *pre-tasse* finale − Investito cumulato = {eur(pre_tax_final)} − {eur(invested_tot)} = **{eur(taxable_gain)}**
        - **Imposta finale (26%)** = {eur(taxable_gain)} × 26% = **{eur(taxes_tot)}**
        - **Valore netto finale** = Valore netto *pre-tasse* finale − Imposta finale = {eur(pre_tax_final)} − {eur(taxes_tot)} = **{eur(net_final)}**


        *TER* e *bollo* sono già sottratti nel corso degli anni, per questo il valore *pre-tasse* è già depurato da questi costi prima dell'applicazione dell'imposta.
        """
        )
    else:
        st.caption("ℹ️ **Nota fiscale**: le tasse (26%) sono applicate solo alla fine sulla plusvalenza realizzata; TER e bollo sono già considerati anno per anno.")


    st.markdown("---")

    # -----------------------------
    # ETF info (quando in modalità ETF)
    # -----------------------------
    if mode == "ETF (media 5 anni)":
        st.subheader("ℹ️ Dettaglio calcolo rendimento da ETF")
        returns_info = st.session_state.get("returns_info", [])
        metric_name = st.session_state.get("metric_name", "Media 5Y")
        if returns_info:
            tmp = pd.DataFrame(returns_info)
            # Riepilogo metrica effettiva usata
            metric_key = "cagr" if metric_name == "CAGR" else "avg_arith"
            # calcola media ponderata netta per trasparenza
            # (serve la tabella dei pesi, qui non la riportiamo: mostriamo solo i valori per ETF)
            if metric_key == "cagr":
                tmp["Metrica usata"] = (tmp["cagr"]*100).map(lambda x: f"{x:.2f}%")
            else:
                tmp["Metrica usata"] = (tmp["avg_arith"]*100).map(lambda x: f"{x:.2f}%")
            tmp.rename(columns={"avg_arith":"Media annua (5Y)", "cagr":"CAGR (≈5Y)", "years_measured":"Anni osservati"}, inplace=True)
            tmp["Media annua (5Y)"] = tmp["Media annua (5Y)"]*100
            tmp["Media annua (5Y)"] = tmp["Media annua (5Y)"].map(lambda x: f"{x:.2f}%")
            tmp["CAGR (≈5Y)"] = tmp["CAGR (≈5Y)"]*100
            tmp["CAGR (≈5Y)"] = tmp["CAGR (≈5Y)"].map(lambda x: f"{x:.2f}%")
            tmp["Anni osservati"] = tmp["Anni osservati"].map(lambda x: f"{x:.0f}")
            st.dataframe(tmp[["ticker", "resolved", "Media annua (5Y)", "CAGR (≈5Y)", "Anni osservati", "Metrica usata"]], width='stretch', hide_index=True)

            st.caption("""
            ℹ️ **Nota**: i rendimenti storici degli ETF sono tipicamente **già al netto del TER**. Qui usiamo
            una **media ponderata** delle metriche scelte (CAGR o Media annua) e ricostruiamo un rendimento **lordo**
            aggiungendo il TER medio ponderato, così da separare le voci di costo (TER, Tasse, Bollo).
            Le **tasse** vengono applicate **solo alla fine** della simulazione.
            """)

    # -----------------------------
    # Chart
    # -----------------------------
    st.markdown("---")
    st.subheader("📈 Andamento nel tempo")
    if PLOTLY_AVAILABLE:
        fig = px.line(
            monthly_df,
            x="Data",
            y=["Investito cumulato", "Valore lordo", "Valore netto"],
            labels={"value":"€", "variable":"Serie"},
            title="Serie cumulative (mensili)",
        )
        # Monte Carlo band (se attivata)
        # try:
        #   mc_enabled
        # except NameError:
        #     mc_enabled = False

        if mc_enabled:
            mu_ann_net = float(st.session_state.get("mu_ann_net", (gross_return_yr - ter_weighted)))
            sigma_ann  = float(st.session_state.get("sigma_ann", 0.0))

            paths = simulate_pac_stochastic(
                years=anni,
                start_monthly=start_monthly,
                annual_increase=annual_up,
                cap_monthly=cap_monthly,
                mu_ann=mu_ann_net,
                sigma_ann=sigma_ann,
                ter_weighted=ter_weighted,
                tax_rate=tax_rate,
                bollo_rate=bollo_rate,
                runs=int(mc_runs),
                seed=int(mc_seed),
            )
            clip_floor, clip_cap = (-0.40, 0.40) if mc_band.startswith("10") else (-0.60, 0.60)
            low_q, high_q = (0.05, 0.95) if mc_band.startswith("5") else (0.10, 0.90)
            p_low  = np.quantile(paths,  low_q, axis=0)
            p_med  = np.quantile(paths,  0.50,  axis=0)
            p_high = np.quantile(paths,  high_q, axis=0)

            years_axis = pd.date_range(monthly_df["Data"].iloc[0].normalize(), periods=anni, freq="Y")
            import plotly.graph_objects as go
            fig.add_trace(go.Scatter(
                x=years_axis, y=p_high,
                line=dict(color="rgba(37,99,235,0.0)"), showlegend=False, hoverinfo="skip"
            ))
            fig.add_trace(go.Scatter(
                x=years_axis, y=p_low,
                line=dict(color="rgba(37,99,235,0.0)"),
                fill='tonexty', fillcolor='rgba(37,99,235,0.18)',
                name=f"Banda confidenza {mc_band}", hovertemplate="~€%{y:,.0f}"
            ))
            fig.add_trace(go.Scatter(
                x=years_axis, y=p_med,
                line=dict(color="rgba(37,99,235,0.9)", width=2, dash="dot"),
                name="Mediana Monte Carlo"
            ))

        fig.update_layout(height=460, legend_title_text="", margin=dict(t=40,l=10,r=10,b=10))
        st.plotly_chart(fig, width='stretch')
    else:
        st.line_chart(monthly_df.set_index("Data")[ ["Investito cumulato", "Valore lordo", "Valore netto"] ])

    # -----------------------------
    # Tables
    # -----------------------------
    st.subheader("📅 Dettaglio annuale")
    pretty_yearly = yearly_df.copy()

    # Elimina colonne non volute
    cols_to_drop = ["Tasse (anno)", "Valore netto (fine anno)"]
    pretty_yearly = pretty_yearly.drop(columns=[c for c in cols_to_drop if c in pretty_yearly.columns])

    money_cols = [c for c in pretty_yearly.columns if c not in ("Anno")]
    for c in money_cols:
        pretty_yearly[c] = pretty_yearly[c].apply(lambda v: eur(v) if pd.notnull(v) else v)

    st.dataframe(pretty_yearly, width='stretch', hide_index=True)


    # Download raw data
    st.download_button(
        label="⬇️ Scarica CSV (dettaglio annuale)",
        data=yearly_df.to_csv(index=False).encode("utf-8"),
        file_name="simulazione_pac_annuale.csv",
        mime="text/csv",
        width='stretch',
    )
else:
    st.info("Configura i parametri nella sidebar e clicca **Esegui simulazione**.")

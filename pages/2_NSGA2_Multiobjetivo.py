"""
pages/2_NSGA2_Multiobjetivo.py
Módulo 2 — NSGA-II Multiobjetivo (DEAP).
Adaptado de: Modulo2_NSGA2_Multiobjetivo.ipynb
"""

import numpy as np
import plotly.express as px
import pandas as pd
import streamlit as st

from utils.nsga2 import ejecutar_nsga2
from utils.data_loader import simular_riqueza, serie_a_dict
from utils.export_excel import df_a_excel_bytes

st.set_page_config(page_title="Módulo 2 · NSGA-II", page_icon="🧬", layout="wide")
st.title("🧬 Módulo 2 — Optimización Multiobjetivo con NSGA-II")

if not st.session_state.get("datos_cargados"):
    st.warning("⚠️ Primero carga los datos desde la página principal (sidebar → 'Cargar Datos').")
    st.stop()

mu = st.session_state["mu"]
cov = st.session_state["cov"]
precios = st.session_state["precios"]
tick_list = st.session_state["tick_list"]
capital = st.session_state["capital"]
frecuencia = st.session_state["frecuencia"]
mu_pop = st.session_state["mu_pop"]
ngen = st.session_state["ngen"]
retornos_simples = precios.pct_change().dropna()

st.caption(f"Población (MU) = {mu_pop} · Generaciones (NGEN) = {ngen} — configurables en el sidebar.")

ejecutar = st.button("▶️ Ejecutar NSGA-II", type="primary")

if ejecutar or "resultados_m2" in st.session_state:
    if ejecutar:
        progress_bar = st.progress(0, text="Iniciando evolución NSGA-II...")

        def _callback(gen, total):
            progress_bar.progress(gen / total, text=f"Generación {gen}/{total}")

        frente, hv_historia = ejecutar_nsga2(mu, cov, mu_pop, ngen, progress_callback=_callback)
        progress_bar.empty()
        st.session_state["frente_nsga2"] = frente
        st.session_state["hv_historia"] = hv_historia
    else:
        frente = st.session_state["frente_nsga2"]
        hv_historia = st.session_state["hv_historia"]

    rets = [r["ret"] for r in frente]
    vols = [r["vol"] for r in frente]
    sharpes = [r["sharpe"] for r in frente]

    idx_conservador = int(np.argmin(vols))
    idx_agresivo = int(np.argmax(rets))
    idx_balanceado = int(np.argmax(sharpes))

    st.success(f"Frente de Pareto calculado: {len(frente)} soluciones no dominadas.")

    # ---- Frontera Pareto vs Markowitz ----
    st.subheader("Frente de Pareto NSGA-II")

    frames_animacion = []
    for i in range(1, len(vols) + 1):
        df_temp = pd.DataFrame({
            "Volatilidad": vols[:i],
            "Retorno": rets[:i],
            "Sharpe": sharpes[:i],
            "Paso": i 
        })
        frames_animacion.append(df_temp)

    df_animacion = pd.concat(frames_animacion)

    rango_x = [min(vols) * 0.95, max(vols) * 1.05]
    rango_y = [min(rets) * 0.95, max(rets) * 1.05]

    fig_pareto = px.scatter(
        df_animacion, 
        x="Volatilidad", 
        y="Retorno", 
        color="Sharpe",
        animation_frame="Paso",
        labels={"Volatilidad": "Volatilidad", "Retorno": "Retorno", "Sharpe": "Sharpe"},
        color_continuous_scale="YlGnBu", 
        title="Frente de Pareto NSGA-II (Aparición Secuencial)",
        range_x=rango_x,
        range_y=rango_y
    )

    fig_pareto.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 75
    fig_pareto.layout.updatemenus[0].buttons[0].args[1]["transition"]["duration"] = 0 

    if "resultados_m1" in st.session_state:
        m1 = st.session_state["resultados_m1"]
        fig_pareto.add_scatter(x=m1["frontera_vols"], y=m1["frontera_objetivos"], mode="lines",
                                name="Frontera Markowitz", line=dict(color="#B3452F", dash="dash"))
    else:
        st.info("Ejecuta primero el Módulo 1 para superponer la frontera de Markowitz.")

    st.plotly_chart(fig_pareto, use_container_width=True)

    # ---- Convergencia ----
    st.subheader("Convergencia del Hypervolumen")
    fig_hv = px.line(x=list(range(1, len(hv_historia) + 1)), y=hv_historia,
                      labels={"x": "Generación", "y": "Hypervolume Indicator"},
                      title="Evolución del Hypervolumen")
    st.plotly_chart(fig_hv, use_container_width=True)

    # ---- 3 portafolios representativos ----
    st.subheader("Portafolios Representativos del Frente de Pareto")
    c1, c2, c3 = st.columns(3)
    with c1:
        fig_p1 = px.pie(names=tick_list, values=frente[idx_conservador]["weights"], hole=0.3,
                         title=f"Conservador (σ={vols[idx_conservador]*100:.1f}%)")
        st.plotly_chart(fig_p1, use_container_width=True)
    with c2:
        fig_p2 = px.pie(names=tick_list, values=frente[idx_balanceado]["weights"], hole=0.3,
                         title=f"Balanceado (Sharpe={sharpes[idx_balanceado]:.2f})")
        st.plotly_chart(fig_p2, use_container_width=True)
    with c3:
        fig_p3 = px.pie(names=tick_list, values=frente[idx_agresivo]["weights"], hole=0.3,
                         title=f"Agresivo (Retorno={rets[idx_agresivo]*100:.1f}%)")
        st.plotly_chart(fig_p3, use_container_width=True)

    # ---- Simulación de riqueza ----
    st.subheader("Simulación de Riqueza — Portafolios GA")
    wealth_conservador = simular_riqueza(retornos_simples, frente[idx_conservador]["weights"], capital, freq_label=frecuencia)
    wealth_balanceado = simular_riqueza(retornos_simples, frente[idx_balanceado]["weights"], capital, freq_label=frecuencia)
    wealth_agresivo = simular_riqueza(retornos_simples, frente[idx_agresivo]["weights"], capital, freq_label=frecuencia)

    import plotly.graph_objects as go
    fig_wealth = go.Figure()
    fig_wealth.add_trace(go.Scatter(x=wealth_conservador.index, y=wealth_conservador.values, name="Conservador"))
    fig_wealth.add_trace(go.Scatter(x=wealth_balanceado.index, y=wealth_balanceado.values, name="Balanceado"))
    fig_wealth.add_trace(go.Scatter(x=wealth_agresivo.index, y=wealth_agresivo.values, name="Agresivo"))
    fig_wealth.update_layout(title="Evolución de Riqueza (GA Rebalanceado)", xaxis_title="Fecha", yaxis_title="Valor del Portafolio (USD)")
    st.plotly_chart(fig_wealth, use_container_width=True)

    # Guardar para Módulo 4
    st.session_state["resultados_m2"] = {
        "tickers": tick_list,
        "frente_pareto": [{"weights": r["weights"].tolist(), "ret": r["ret"], "vol": r["vol"], "sharpe": r["sharpe"]} for r in frente],
        "hv_historia": hv_historia,
        "idx_conservador": idx_conservador, "idx_balanceado": idx_balanceado, "idx_agresivo": idx_agresivo,
        "wealth_conservador": serie_a_dict(wealth_conservador),
        "wealth_balanceado": serie_a_dict(wealth_balanceado),
        "wealth_agresivo": serie_a_dict(wealth_agresivo),
        "mu_pop": mu_pop, "ngen": ngen,
    }

    # ---- Descarga ----
    st.markdown("---")
    df_pareto = pd.DataFrame([
        {"Ticker_" + t: r["weights"][i] for i, t in enumerate(tick_list)} | {"Retorno": r["ret"], "Volatilidad": r["vol"], "Sharpe": r["sharpe"]}
        for r in frente
    ])
    st.download_button(
        "⬇️ Descargar frente de Pareto completo (Excel)",
        data=df_a_excel_bytes({"Frente_Pareto": df_pareto}),
        file_name="frente_pareto_nsga2.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
else:
    st.info("Presiona **'Ejecutar NSGA-II'** para correr el algoritmo genético con los parámetros del sidebar.")

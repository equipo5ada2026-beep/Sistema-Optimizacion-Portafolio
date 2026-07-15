"""
pages/3_DP_Rebalanceo.py
Módulo 3 — Rebalanceo óptimo vía Programación Dinámica (Bellman).
Adaptado de: Modulo3_DP_Rebalanceo.ipynb
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.markowitz import rendimiento_portafolio, optimizar_max_sharpe
from utils.dp_rebalanceo import ejecutar_dp_rebalanceo, extraer_decisiones, simular_dp_forward
from utils.export_excel import df_a_excel_bytes

st.set_page_config(page_title="Módulo 3 · DP Rebalanceo", page_icon="🧮", layout="wide")
st.title("🧮 Módulo 3 — Rebalanceo Óptimo vía Programación Dinámica")

if not st.session_state.get("datos_cargados"):
    st.warning("⚠️ Primero carga los datos desde la página principal (sidebar → 'Cargar Datos').")
    st.stop()

mu = st.session_state["mu"]
cov = st.session_state["cov"]
log_returns = st.session_state["log_returns"]
capital = st.session_state["capital"]
lambda_tc = st.session_state["lambda_tc"]
horizonte_dp = st.session_state["horizonte_dp"]

# Sliders específicos del módulo (grid step y aversión al riesgo, además de los del sidebar)
col_a, col_b = st.columns(2)
grid_step = col_a.slider("Paso de grilla (Δw)", 0.02, 0.20, 0.05, step=0.01)
aversion_riesgo = col_b.slider("Aversión al riesgo (λ)", 1.0, 10.0, 3.0, step=0.5)

costo_trans_pct = lambda_tc * 100  # el notebook usa % (ej. 0.1 = 0.1%), el sidebar usa fracción

with st.spinner("Resolviendo backward induction (Bellman)..."):
    w_tangente = optimizar_max_sharpe(mu, cov)
    mu_p, sigma_p = rendimiento_portafolio(w_tangente, mu.values, cov.values)

    grid, J, politica, matriz_costos = ejecutar_dp_rebalanceo(
        mu_p, sigma_p, costo_trans_pct, horizonte_dp, grid_step, aversion_riesgo
    )

    port_log_ret = pd.Series(log_returns.values @ w_tangente, index=log_returns.index)
    monthly_log = port_log_ret.resample("ME").sum()
    retornos_periodicos = np.exp(monthly_log) - 1

    fechas_dp, wealth_dp, wealth_bh_dp, wealth_full_dp = simular_dp_forward(
        grid, politica, retornos_periodicos, costo_trans_pct, capital, horizonte_dp
    )

st.caption(f"Portafolio tangente → retorno anual: {mu_p*100:.2f}% · volatilidad anual: {sigma_p*100:.2f}%")

# ---- Métricas ----
c1, c2, c3 = st.columns(3)
c1.metric("Riqueza Final — DP", f"${wealth_dp[-1]:,.0f}")
c2.metric("Riqueza Final — Buy & Hold", f"${wealth_bh_dp[-1]:,.0f}")
c3.metric("Riqueza Final — Siempre Rebalanceado", f"${wealth_full_dp[-1]:,.0f}")

st.markdown("---")

# ---- Política óptima ----
st.subheader("Política Óptima de Rebalanceo")
df_acciones = extraer_decisiones(grid, politica, horizonte_dp)
st.dataframe(df_acciones, use_container_width=True)

# ---- Heatmap ----
st.subheader("Matriz de Costos de Transición (Heatmap)")
grid_labels = [f"{w:.2f}" for w in grid]
fig_heat = px.imshow(matriz_costos,
                      labels=dict(x="Estado Destino (w_t)", y="Estado Origen (w_t-1)", color="Costo Ajuste"),
                      x=grid_labels, y=grid_labels, title="Matriz de Costos de Transición")
st.plotly_chart(fig_heat, use_container_width=True)

# ---- Evolución de riqueza ----
st.subheader("Evolución de Riqueza — 3 Estrategias")
fig_dp = go.Figure()
fig_dp.add_trace(go.Scatter(x=list(range(len(wealth_dp))), y=wealth_dp,
                             name="Estrategia DP (Óptima)", line=dict(color="#7C9473", width=3)))
fig_dp.add_trace(go.Scatter(x=list(range(len(wealth_bh_dp))), y=wealth_bh_dp,
                             name="Buy & Hold", line=dict(color="#B3452F", dash="dash")))
fig_dp.add_trace(go.Scatter(x=list(range(len(wealth_full_dp))), y=wealth_full_dp,
                             name="Siempre Rebalanceado (w=1)", line=dict(color="#3D4F4A", dash="dot")))
fig_dp.update_layout(title="Evolución de Riqueza - Programación Dinámica",
                      xaxis_title="Periodo", yaxis_title="Valor del Portafolio (USD)")
st.plotly_chart(fig_dp, use_container_width=True)

# Guardar para Módulo 4
st.session_state["resultados_m3"] = {
    "grid": grid.tolist(), "mu_p": mu_p, "sigma_p": sigma_p,
    "costo_transaccion_pct": costo_trans_pct, "horizonte": horizonte_dp, "grid_step": grid_step,
    "wealth_dp": {"periodos": list(range(len(wealth_dp))), "valores": wealth_dp},
    "wealth_bh_dp": {"periodos": list(range(len(wealth_bh_dp))), "valores": wealth_bh_dp},
    "wealth_full_dp": {"periodos": list(range(len(wealth_full_dp))), "valores": wealth_full_dp},
    "capital_inicial": capital,
}

# ---- Descarga ----
st.markdown("---")
df_sim = pd.DataFrame({
    "Periodo": list(range(len(wealth_dp))),
    "Wealth_DP": wealth_dp, "Wealth_BuyHold": wealth_bh_dp, "Wealth_SiempreRebalanceado": wealth_full_dp,
})
st.download_button(
    "⬇️ Descargar simulación DP (Excel)",
    data=df_a_excel_bytes({"Simulacion_DP": df_sim, "Politica_Optima": df_acciones}),
    file_name="simulacion_dp_rebalanceo.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

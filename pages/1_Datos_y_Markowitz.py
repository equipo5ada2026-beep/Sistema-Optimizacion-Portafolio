"""
pages/1_Datos_y_Markowitz.py
Módulo 1 — Frontera eficiente de Markowitz, portafolios óptimos y backtesting.
Adaptado de: Modulo1_Datos_Markowitz.ipynb
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.markowitz import rendimiento_portafolio, optimizar_max_sharpe, optimizar_min_varianza, frontera_eficiente
from utils.data_loader import simular_riqueza, serie_a_dict
from utils.export_excel import df_a_excel_bytes

st.set_page_config(page_title="Módulo 1 · Markowitz", page_icon="📊", layout="wide")
st.title("📊 Módulo 1 — Datos y Optimización de Markowitz")

if not st.session_state.get("datos_cargados"):
    st.warning("⚠️ Primero carga los datos desde la página principal (sidebar → 'Cargar Datos').")
    st.stop()

mu = st.session_state["mu"]
cov = st.session_state["cov"]
precios = st.session_state["precios"]
tick_list = st.session_state["tick_list"]
capital = st.session_state["capital"]
frecuencia = st.session_state["frecuencia"]
retornos_simples = precios.pct_change().dropna()

with st.spinner("Optimizando portafolio..."):
    w_sharpe = optimizar_max_sharpe(mu, cov)
    w_minvar = optimizar_min_varianza(mu, cov)

    ret_s, vol_s = rendimiento_portafolio(w_sharpe, mu.values, cov.values)
    ret_mv, vol_mv = rendimiento_portafolio(w_minvar, mu.values, cov.values)
    sharpe_s = ret_s / vol_s if vol_s > 0 else np.nan

    # Sortino del portafolio de máximo Sharpe (a partir de su serie de retornos)
    ret_port_diaria = retornos_simples @ w_sharpe
    downside = ret_port_diaria[ret_port_diaria < 0].std()
    sortino_s = (ret_port_diaria.mean() * 252) / (downside * np.sqrt(252)) if downside and downside > 0 else np.nan

    objetivos, vols_frontera, _ = frontera_eficiente(mu, cov, n_puntos=200)

# ---- Métricas ----
c1, c2, c3 = st.columns(3)
c1.metric("Sharpe Ratio", f"{sharpe_s:.2f}")
c2.metric("Sortino Ratio", f"{sortino_s:.2f}" if pd.notna(sortino_s) else "N/D")
c3.metric("Volatilidad anual", f"{vol_s*100:.2f}%")

st.markdown("---")

# ---- Frontera eficiente ----
st.subheader("Frontera Eficiente de Markowitz")
fig_ef = px.line(x=vols_frontera, y=objetivos,
                  labels={"x": "Volatilidad (Riesgo)", "y": "Retorno Esperado"},
                  title="Frontera Eficiente de Markowitz")
fig_ef.add_scatter(x=[vol_s], y=[ret_s], mode="markers", name="Máximo Sharpe",
                    marker=dict(size=13, color="#C4622D"))
fig_ef.add_scatter(x=[vol_mv], y=[ret_mv], mode="markers", name="Mínima Varianza",
                    marker=dict(size=13, color="#3D4F4A"))
vol_activos = np.sqrt(np.diag(cov.values))
fig_ef.add_scatter(x=vol_activos, y=mu.values, mode="markers+text", name="Activos",
                    text=tick_list, textposition="top center")
st.plotly_chart(fig_ef, use_container_width=True)

# ---- Composición ----
st.subheader("Composición del Portafolio Óptimo (Máximo Sharpe)")
col_pie1, col_pie2 = st.columns(2)
with col_pie1:
    fig_pie_sharpe = px.pie(names=tick_list, values=w_sharpe, title="Asignación — Máximo Sharpe",
                             color_discrete_sequence=px.colors.sequential.YlGnBu)
    st.plotly_chart(fig_pie_sharpe, use_container_width=True)
with col_pie2:
    fig_pie_minvar = px.pie(names=tick_list, values=w_minvar, title="Asignación — Mínima Varianza",
                             color_discrete_sequence=px.colors.sequential.Sunset)
    st.plotly_chart(fig_pie_minvar, use_container_width=True)

st.markdown("---")

# ---- Simulación de riqueza ----
st.subheader("Simulación de Riqueza (Backtesting)")
w_eq = np.repeat(1.0 / len(tick_list), len(tick_list))
wealth_bh = simular_riqueza(retornos_simples, w_sharpe, capital, freq_label=None)
wealth_mkw = simular_riqueza(retornos_simples, w_sharpe, capital, freq_label=frecuencia)
wealth_eq = simular_riqueza(retornos_simples, w_eq, capital, freq_label=frecuencia)

def _figura_riqueza_animada_m1(wealth_bh, wealth_mkw, wealth_eq, freq_label):
    n_frames = min(60, len(wealth_bh))
    indices_frame = np.linspace(0, len(wealth_bh) - 1, n_frames, dtype=int)
    
    y_min = min(wealth_bh.min(), wealth_mkw.min(), wealth_eq.min())
    y_max = max(wealth_bh.max(), wealth_mkw.max(), wealth_eq.max())
    padding_y = max((y_max - y_min) * 0.08, 1.0)
    
    figura = go.Figure()
    
    figura.add_trace(go.Scatter(x=wealth_bh.index[:1], y=wealth_bh.values[:1],
                                name="Buy & Hold (Máx. Sharpe)", line=dict(color="#B3452F", dash="dash")))
    figura.add_trace(go.Scatter(x=wealth_mkw.index[:1], y=wealth_mkw.values[:1],
                                name=f"Markowitz Rebalanceado ({freq_label})", line=dict(color="#1F3864", width=3)))
    figura.add_trace(go.Scatter(x=wealth_eq.index[:1], y=wealth_eq.values[:1],
                                name="Equiponderado", line=dict(color="#C5961A", dash="dot")))
    
    frames = []
    pasos = []
    
    for i, idx in enumerate(indices_frame):
        frames.append(go.Frame(
            name=str(i),
            data=[
                go.Scatter(x=wealth_bh.index[:idx+1], y=wealth_bh.values[:idx+1]),
                go.Scatter(x=wealth_mkw.index[:idx+1], y=wealth_mkw.values[:idx+1]),
                go.Scatter(x=wealth_eq.index[:idx+1], y=wealth_eq.values[:idx+1])
            ]
        ))
        
        pasos.append(
            {
                "label": wealth_bh.index[idx].strftime("%Y-%m") if hasattr(wealth_bh.index, 'strftime') else str(i),
                "method": "animate",
                "args": [
                    [str(i)],
                    {"mode": "immediate", "frame": {"duration": 100, "redraw": True}, "transition": {"duration": 0}}
                ]
            }
        )
        
    figura.frames = frames
    
    figura.update_layout(
        title="Evolución de Riqueza — Animación temporal",
        xaxis={"title": "Fecha", "range": [wealth_bh.index[0], wealth_bh.index[-1]]},
        yaxis={"title": "Valor del Portafolio (USD)", "range": [y_min - padding_y, y_max + padding_y]},
        margin={"t": 120},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.08,
                "y": 0,
                "xanchor": "right",
                "yanchor": "top",
                "pad": {"r": 10, "t": 70},
                "direction": "left",
                "buttons": [
                    {
                        "label": "▶ Reproducir",
                        "method": "animate",
                        "args": [None, {"fromcurrent": True, "frame": {"duration": 100, "redraw": True}, "transition": {"duration": 0}}]
                    },
                    {
                        "label": "⏸ Pausar",
                        "method": "animate",
                        "args": [[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": False}, "transition": {"duration": 0}}]
                    }
                ]
            }
        ],
        sliders=[{"active": 0, "x": 0.1, "len": 0.9, "currentvalue": {"prefix": "Hasta: "}, "pad": {"t": 55}, "steps": pasos}]
    )
    return figura

fig_wealth = _figura_riqueza_animada_m1(wealth_bh, wealth_mkw, wealth_eq, frecuencia)
st.plotly_chart(fig_wealth, use_container_width=True)

# Guardar resultados del módulo para el Módulo 4 (Comparación)
st.session_state["resultados_m1"] = {
    "tickers": tick_list,
    "w_sharpe": w_sharpe.tolist(), "w_minvar": w_minvar.tolist(),
    "ret_sharpe": ret_s, "vol_sharpe": vol_s, "sharpe_ratio": sharpe_s,
    "ret_minvar": ret_mv, "vol_minvar": vol_mv,
    "frontera_objetivos": objetivos.tolist(), "frontera_vols": vols_frontera.tolist(),
    "wealth_bh": serie_a_dict(wealth_bh),
    "wealth_mkw": serie_a_dict(wealth_mkw),
    "wealth_eq": serie_a_dict(wealth_eq),
    "capital_inicial": capital, "frequency": frecuencia,
}

# ---- Descarga ----
st.markdown("---")
df_pesos = pd.DataFrame({
    "Ticker": tick_list,
    "Peso Máximo Sharpe": w_sharpe,
    "Peso Mínima Varianza": w_minvar,
})
st.download_button(
    "⬇️ Descargar pesos del portafolio (Excel)",
    data=df_a_excel_bytes({"Pesos_Portafolio": df_pesos}),
    file_name="pesos_portafolio_markowitz.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

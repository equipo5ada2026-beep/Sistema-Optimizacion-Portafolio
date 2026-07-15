"""
pages/4_Comparacion.py
Módulo 4 — Comparación Cruzada de todos los métodos.
Adaptado de: Modulo4_Comparacion.ipynb
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.metrics import metricas_desempeno
from utils.export_excel import df_a_excel_bytes

st.set_page_config(page_title="Módulo 4 · Comparación", page_icon="🏆", layout="wide")
st.title("🏆 Módulo 4 — Comparación Cruzada de Métodos")

faltantes = [m for m in ["resultados_m1", "resultados_m2", "resultados_m3"] if m not in st.session_state]
if faltantes:
    nombres = {"resultados_m1": "Módulo 1 (Markowitz)", "resultados_m2": "Módulo 2 (NSGA-II)", "resultados_m3": "Módulo 3 (DP)"}
    st.warning(
        "⚠️ Para ver la comparación cruzada debes ejecutar primero: "
        + ", ".join(nombres[m] for m in faltantes)
    )
    st.stop()

m1 = st.session_state["resultados_m1"]
m2 = st.session_state["resultados_m2"]
m3 = st.session_state["resultados_m3"]

st.success("Módulos cargados correctamente: M1 (Markowitz), M2 (NSGA-II), M3 (Programación Dinámica).")


def dict_a_serie(d, usar_fechas=True):
    if usar_fechas:
        idx = pd.to_datetime(d["fechas"])
    else:
        idx = d.get("periodos", list(range(len(d["valores"]))))
    return pd.Series(d["valores"], index=idx)


curvas = {
    "Markowitz - Buy & Hold": dict_a_serie(m1["wealth_bh"]),
    "Markowitz - Rebalanceado": dict_a_serie(m1["wealth_mkw"]),
    "Equiponderado": dict_a_serie(m1["wealth_eq"]),
    "NSGA-II - Conservador": dict_a_serie(m2["wealth_conservador"]),
    "NSGA-II - Balanceado": dict_a_serie(m2["wealth_balanceado"]),
    "NSGA-II - Agresivo": dict_a_serie(m2["wealth_agresivo"]),
    "Programacion Dinamica": dict_a_serie(m3["wealth_dp"], usar_fechas=False),
}

FRECUENCIA_ANUAL = {
    "Markowitz - Buy & Hold": 252, "Markowitz - Rebalanceado": 252, "Equiponderado": 252,
    "NSGA-II - Conservador": 252, "NSGA-II - Balanceado": 252, "NSGA-II - Agresivo": 252,
    "Programacion Dinamica": 12,
}

st.caption(
    "Nota: la curva de Programación Dinámica usa periodicidad mensual (12/año); "
    "las demás usan periodicidad diaria (252/año). Tasa libre de riesgo asumida = 0%."
)

# ---- Tabla de métricas ----
filas = []
for nombre, serie in curvas.items():
    met = metricas_desempeno(serie.values, periodos_por_anio=FRECUENCIA_ANUAL[nombre])
    if met:
        fila = {"Estrategia": nombre}
        fila.update(met)
        filas.append(fila)

df_comparacion = pd.DataFrame(filas)

st.subheader("Tabla Resumen de Métricas")
st.dataframe(df_comparacion, use_container_width=True)

st.markdown("---")

# ---- Gráfico de barras comparativo ----
st.subheader("Comparación de Métricas Clave")
metric_sel = st.selectbox("Métrica a graficar", ["Sharpe Ratio", "Riqueza Final (USD)", "Max Drawdown (%)"])
fig_bar = px.bar(df_comparacion.sort_values(metric_sel, ascending=False),
                  x="Estrategia", y=metric_sel, color="Estrategia", title=f"{metric_sel} por Estrategia")
fig_bar.update_layout(showlegend=False)
st.plotly_chart(fig_bar, use_container_width=True)

# ---- Evolución superpuesta (base 100) ----
st.subheader("Evolución de Riqueza Normalizada (Base 100) — 7 Estrategias")
fig_comp = go.Figure()
for nombre, serie in curvas.items():
    es_temporal = not isinstance(serie.index[0], (int, np.integer))
    x_vals = serie.index if es_temporal else list(range(len(serie)))
    y_norm = serie.values / serie.values[0] * 100
    fig_comp.add_trace(go.Scatter(x=x_vals, y=y_norm, name=nombre, mode="lines"))
fig_comp.update_layout(title="Evolución de Riqueza Normalizada (Base 100) - 7 Estrategias",
                        xaxis_title="Tiempo", yaxis_title="Riqueza Normalizada")
st.plotly_chart(fig_comp, use_container_width=True)

st.markdown("---")

# ---- Ranking final ----
st.subheader("🏆 Ranking Final por Sharpe Ratio")
df_ranking = df_comparacion.sort_values("Sharpe Ratio", ascending=False).reset_index(drop=True)
df_ranking.index += 1
st.dataframe(df_ranking, use_container_width=True)

mejor = df_ranking.iloc[0]
st.info(f"🥇 **Mejor estrategia por Sharpe Ratio:** {mejor['Estrategia']} (Sharpe = {mejor['Sharpe Ratio']})")

df_ranking_wealth = df_comparacion.sort_values("Riqueza Final (USD)", ascending=False).reset_index(drop=True)
df_ranking_wealth.index += 1
mejor_wealth = df_ranking_wealth.iloc[0]
st.info(f"💰 **Mejor estrategia por Riqueza Final:** {mejor_wealth['Estrategia']} (${mejor_wealth['Riqueza Final (USD)']:,.0f})")

# ---- Descarga: reporte completo multi-hoja ----
st.markdown("---")
hojas = {"Tabla_Comparacion": df_comparacion, "Ranking_Sharpe": df_ranking, "Ranking_Riqueza": df_ranking_wealth}
for nombre, serie in curvas.items():
    hojas[nombre.replace(" ", "_")[:31]] = serie.reset_index().rename(columns={"index": "Fecha/Periodo", 0: "Riqueza"})

st.download_button(
    "⬇️ Descargar Reporte Completo (Excel, multi-hoja)",
    data=df_a_excel_bytes(hojas),
    file_name="reporte_comparacion_completo.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

"""
pages/3_DP_Rebalanceo.py
Módulo 3 — Rebalanceo óptimo vía Programación Dinámica (Bellman).
Adaptado de: Modulo3_DP_Rebalanceo.ipynb
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.markowitz import rendimiento_portafolio, optimizar_max_sharpe
from utils.dp_rebalanceo import ejecutar_dp_rebalanceo, extraer_decisiones, simular_dp_forward
from utils.export_excel import df_a_excel_bytes


st.set_page_config(
    page_title="Módulo 3 · DP Rebalanceo",
    page_icon="🧮",
    layout="wide",
)

st.title(
    "🧮 Módulo 3 — Rebalanceo Óptimo vía Programación Dinámica"
)


if not st.session_state.get("datos_cargados"):
    st.warning(
        "⚠️ Primero carga los datos desde la página principal "
        "(sidebar → 'Cargar Datos')."
    )
    st.stop()


mu = st.session_state["mu"]
cov = st.session_state["cov"]
log_returns = st.session_state["log_returns"]
capital = st.session_state["capital"]
lambda_tc = st.session_state["lambda_tc"]
horizonte_dp = st.session_state["horizonte_dp"]


# Sliders específicos del módulo.
col_a, col_b = st.columns(2)

grid_step = col_a.slider(
    "Paso de grilla (Δw)",
    0.02,
    0.20,
    0.05,
    step=0.01,
)

aversion_riesgo = col_b.slider(
    "Aversión al riesgo (λ)",
    1.0,
    10.0,
    3.0,
    step=0.5,
)

# El notebook usa porcentaje; el sidebar almacena una fracción.
costo_trans_pct = lambda_tc * 100


@st.cache_data(show_spinner=False)
def _resolver_modulo_dp(
    mu_values,
    cov_values,
    log_returns_values,
    log_returns_index,
    costo_transaccion,
    horizonte,
    paso_grilla,
    aversion,
    capital_inicial,
):
    """Ejecuta el cálculo principal y permite reutilizarlo al mover la página."""
    mu_serie = pd.Series(mu_values)
    cov_df = pd.DataFrame(cov_values)

    w_tangente = optimizar_max_sharpe(mu_serie, cov_df)
    mu_p, sigma_p = rendimiento_portafolio(
        w_tangente,
        mu_serie.values,
        cov_df.values,
    )

    grid, J, politica, matriz_costos = ejecutar_dp_rebalanceo(
        mu_p,
        sigma_p,
        costo_transaccion,
        horizonte,
        paso_grilla,
        aversion,
    )

    log_returns_df = pd.DataFrame(
        log_returns_values,
        index=pd.to_datetime(log_returns_index),
    )

    port_log_ret = pd.Series(
        log_returns_df.values @ w_tangente,
        index=log_returns_df.index,
    )

    monthly_log = port_log_ret.resample("ME").sum()
    retornos_periodicos = np.exp(monthly_log) - 1

    fechas_dp, wealth_dp, wealth_bh_dp, wealth_full_dp = simular_dp_forward(
        grid,
        politica,
        retornos_periodicos,
        costo_transaccion,
        capital_inicial,
        horizonte,
    )

    return (
        w_tangente,
        mu_p,
        sigma_p,
        grid,
        J,
        politica,
        matriz_costos,
        fechas_dp,
        wealth_dp,
        wealth_bh_dp,
        wealth_full_dp,
    )


with st.spinner("Resolviendo backward induction (Bellman)..."):
    (
        w_tangente,
        mu_p,
        sigma_p,
        grid,
        J,
        politica,
        matriz_costos,
        fechas_dp,
        wealth_dp,
        wealth_bh_dp,
        wealth_full_dp,
    ) = _resolver_modulo_dp(
        mu.values,
        cov.values,
        log_returns.values,
        log_returns.index.astype(str).tolist(),
        costo_trans_pct,
        horizonte_dp,
        grid_step,
        aversion_riesgo,
        capital,
    )


st.caption(
    f"Portafolio tangente → retorno anual: {mu_p * 100:.2f}% · "
    f"volatilidad anual: {sigma_p * 100:.2f}%"
)


# -----------------------------------------------------------------------------
# Funciones de animación
# -----------------------------------------------------------------------------


def _figura_heatmap_animada(matriz, etiquetas):
    """
    Revela la matriz fila por fila.

    Cada fila representa los costos de pasar desde un estado de origen hacia
    todos los estados destino posibles.
    """
    matriz = np.asarray(matriz, dtype=float)
    n_filas, n_columnas = matriz.shape

    z_min = float(np.nanmin(matriz))
    z_max = float(np.nanmax(matriz))

    if np.isclose(z_min, z_max):
        z_max = z_min + 1e-12

    def matriz_parcial(filas_visibles):
        parcial = np.full_like(matriz, np.nan, dtype=float)
        parcial[:filas_visibles, :] = matriz[:filas_visibles, :]
        return parcial

    figura = go.Figure(
        data=[
            go.Heatmap(
                z=matriz_parcial(1),
                x=etiquetas,
                y=etiquetas,
                colorscale="Blues",
                zmin=z_min,
                zmax=z_max,
                colorbar={"title": "Costo ajuste"},
                hovertemplate=(
                    "Estado origen=%{y}<br>"
                    "Estado destino=%{x}<br>"
                    "Costo=%{z:.6f}"
                    "<extra></extra>"
                ),
            )
        ]
    )

    frames = []

    for fila in range(1, n_filas + 1):
        frames.append(
            go.Frame(
                name=str(fila),
                data=[
                    go.Heatmap(
                        z=matriz_parcial(fila),
                        x=etiquetas,
                        y=etiquetas,
                        colorscale="Blues",
                        zmin=z_min,
                        zmax=z_max,
                        showscale=True,
                        colorbar={"title": "Costo ajuste"},
                        hovertemplate=(
                            "Estado origen=%{y}<br>"
                            "Estado destino=%{x}<br>"
                            "Costo=%{z:.6f}"
                            "<extra></extra>"
                        ),
                    )
                ],
            )
        )

    figura.frames = frames

    pasos = []

    for fila in range(1, n_filas + 1):
        pasos.append(
            {
                "label": str(fila),
                "method": "animate",
                "args": [
                    [str(fila)],
                    {
                        "mode": "immediate",
                        "frame": {
                            "duration": 220,
                            "redraw": True,
                        },
                        "transition": {"duration": 0},
                    },
                ],
            }
        )

    figura.update_layout(
        title=(
            "Matriz de Costos de Transición — "
            "construcción por estado de origen"
        ),
        xaxis={
            "title": "Estado Destino (w_t)",
            "type": "category",
        },
        yaxis={
            "title": "Estado Origen (w_t-1)",
            "type": "category",
            "autorange": "reversed",
        },
        height=650,
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0,
                "y": 1.14,
                "direction": "left",
                "buttons": [
                    {
                        "label": "▶ Reproducir",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {
                                    "duration": 220,
                                    "redraw": True,
                                },
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "⏸ Pausar",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {
                                    "duration": 0,
                                    "redraw": False,
                                },
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {
                    "prefix": "Filas calculadas: ",
                },
                "pad": {"t": 55},
                "steps": pasos,
            }
        ],
    )

    return figura



def _figura_riqueza_animada(
    riqueza_dp,
    riqueza_bh,
    riqueza_full,
):
    """Dibuja las tres estrategias periodo por periodo."""
    riqueza_dp = list(riqueza_dp)
    riqueza_bh = list(riqueza_bh)
    riqueza_full = list(riqueza_full)

    n_periodos = min(
        len(riqueza_dp),
        len(riqueza_bh),
        len(riqueza_full),
    )

    periodos = list(range(n_periodos))

    todos_valores = (
        riqueza_dp[:n_periodos]
        + riqueza_bh[:n_periodos]
        + riqueza_full[:n_periodos]
    )

    y_min = min(todos_valores)
    y_max = max(todos_valores)
    padding_y = max((y_max - y_min) * 0.08, 1.0)

    estilos = [
        {
            "name": "Estrategia DP (Óptima)",
            "line": {"color": "#7C9473", "width": 3},
        },
        {
            "name": "Buy & Hold",
            "line": {"color": "#B3452F", "dash": "dash"},
        },
        {
            "name": "Siempre Rebalanceado (w=1)",
            "line": {"color": "#3D4F4A", "dash": "dot"},
        },
    ]

    series = [
        riqueza_dp[:n_periodos],
        riqueza_bh[:n_periodos],
        riqueza_full[:n_periodos],
    ]

    figura = go.Figure()

    for valores, estilo in zip(series, estilos):
        figura.add_trace(
            go.Scatter(
                x=periodos[:1],
                y=valores[:1],
                mode="lines+markers",
                name=estilo["name"],
                line=estilo["line"],
                marker={"size": 7},
                hovertemplate=(
                    "Periodo=%{x}<br>"
                    "Riqueza=$%{y:,.2f}"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    frames = []

    for indice in range(n_periodos):
        datos_frame = []

        for valores, estilo in zip(series, estilos):
            datos_frame.append(
                go.Scatter(
                    x=periodos[: indice + 1],
                    y=valores[: indice + 1],
                    mode="lines+markers",
                    name=estilo["name"],
                    line=estilo["line"],
                    marker={"size": 7},
                    hovertemplate=(
                        "Periodo=%{x}<br>"
                        "Riqueza=$%{y:,.2f}"
                        "<extra>%{fullData.name}</extra>"
                    ),
                )
            )

        frames.append(
            go.Frame(
                name=str(indice),
                data=datos_frame,
            )
        )

    figura.frames = frames

    pasos = []

    for indice in range(n_periodos):
        pasos.append(
            {
                "label": str(indice),
                "method": "animate",
                "args": [
                    [str(indice)],
                    {
                        "mode": "immediate",
                        "frame": {
                            "duration": 450,
                            "redraw": True,
                        },
                        "transition": {"duration": 0},
                    },
                ],
            }
        )

    figura.update_layout(
        title="Evolución de Riqueza — construcción periodo por periodo",
        xaxis={
            "title": "Periodo",
            "range": [0, max(periodos) if periodos else 0],
            "dtick": 1,
        },
        yaxis={
            "title": "Valor del Portafolio (USD)",
            "range": [y_min - padding_y, y_max + padding_y],
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0,
                "y": 1.16,
                "direction": "left",
                "buttons": [
                    {
                        "label": "▶ Reproducir",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "fromcurrent": True,
                                "frame": {
                                    "duration": 450,
                                    "redraw": True,
                                },
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "⏸ Pausar",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {
                                    "duration": 0,
                                    "redraw": False,
                                },
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "currentvalue": {
                    "prefix": "Periodo mostrado: ",
                },
                "pad": {"t": 55},
                "steps": pasos,
            }
        ],
    )

    return figura


# -----------------------------------------------------------------------------
# Resultados
# -----------------------------------------------------------------------------

c1, c2, c3 = st.columns(3)

c1.metric(
    "Riqueza Final — DP",
    f"${wealth_dp[-1]:,.0f}",
)

c2.metric(
    "Riqueza Final — Buy & Hold",
    f"${wealth_bh_dp[-1]:,.0f}",
)

c3.metric(
    "Riqueza Final — Siempre Rebalanceado",
    f"${wealth_full_dp[-1]:,.0f}",
)

st.markdown("---")


st.subheader("Política Óptima de Rebalanceo")

df_acciones = extraer_decisiones(
    grid,
    politica,
    horizonte_dp,
)

st.dataframe(
    df_acciones,
    use_container_width=True,
)


st.subheader("Matriz de Costos de Transición (Heatmap)")

st.info(
    "La matriz se revela fila por fila. Cada fila corresponde a un estado "
    "de origen y muestra el costo de transición hacia todos los destinos."
)

grid_labels = [
    f"{peso:.2f}"
    for peso in grid
]

fig_heat = _figura_heatmap_animada(
    matriz_costos,
    grid_labels,
)

st.plotly_chart(
    fig_heat,
    use_container_width=True,
)


st.subheader("Evolución de Riqueza — 3 Estrategias")

fig_dp = _figura_riqueza_animada(
    wealth_dp,
    wealth_bh_dp,
    wealth_full_dp,
)

st.plotly_chart(
    fig_dp,
    use_container_width=True,
)


# Guardar para el Módulo 4.
st.session_state["resultados_m3"] = {
    "grid": grid.tolist(),
    "mu_p": mu_p,
    "sigma_p": sigma_p,
    "costo_transaccion_pct": costo_trans_pct,
    "horizonte": horizonte_dp,
    "grid_step": grid_step,
    "wealth_dp": {
        "periodos": list(range(len(wealth_dp))),
        "valores": wealth_dp,
    },
    "wealth_bh_dp": {
        "periodos": list(range(len(wealth_bh_dp))),
        "valores": wealth_bh_dp,
    },
    "wealth_full_dp": {
        "periodos": list(range(len(wealth_full_dp))),
        "valores": wealth_full_dp,
    },
    "capital_inicial": capital,
}


st.markdown("---")


df_sim = pd.DataFrame(
    {
        "Periodo": list(range(len(wealth_dp))),
        "Wealth_DP": wealth_dp,
        "Wealth_BuyHold": wealth_bh_dp,
        "Wealth_SiempreRebalanceado": wealth_full_dp,
    }
)

st.download_button(
    "⬇️ Descargar simulación DP (Excel)",
    data=df_a_excel_bytes(
        {
            "Simulacion_DP": df_sim,
            "Politica_Optima": df_acciones,
        }
    ),
    file_name="simulacion_dp_rebalanceo.xlsx",
    mime=(
        "application/vnd.openxmlformats-"
        "officedocument.spreadsheetml.sheet"
    ),
)

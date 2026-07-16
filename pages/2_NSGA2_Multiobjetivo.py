"""
pages/2_NSGA2_Multiobjetivo.py
Módulo 2 — NSGA-II Multiobjetivo.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.data_loader import (
    serie_a_dict,
    simular_riqueza,
)
from utils.export_excel import df_a_excel_bytes
from utils.nsga2 import ejecutar_nsga2


st.set_page_config(
    page_title="Módulo 2 · NSGA-II",
    page_icon="🧬",
    layout="wide",
)

st.title(
    "🧬 Módulo 2 — Optimización Multiobjetivo con NSGA-II"
)


if not st.session_state.get("datos_cargados"):
    st.warning(
        "⚠️ Primero carga los datos desde la página principal "
        "(sidebar → 'Cargar Datos')."
    )
    st.stop()


mu = st.session_state["mu"]
cov = st.session_state["cov"]
precios = st.session_state["precios"]
tick_list = st.session_state["tick_list"]
capital = st.session_state["capital"]
frecuencia = st.session_state["frecuencia"]
mu_pop = st.session_state["mu_pop"]
ngen = st.session_state["ngen"]

retornos_simples = (
    precios
    .pct_change()
    .dropna()
)


st.caption(
    f"Población (MU) = {mu_pop} · "
    f"Generaciones (NGEN) = {ngen} — "
    "configurables en el sidebar."
)


def _figura_pareto_animada(
    historial_generaciones,
    markowitz=None,
):
    """
    Construye una animación que muestra la evolución de la población
    y del frente de Pareto en cada generación.
    """
    registros = [
        punto
        for generacion in historial_generaciones
        for punto in generacion
    ]

    todas_volatilidades = [
        punto["vol"]
        for punto in registros
    ]

    todos_retornos = [
        punto["ret"]
        for punto in registros
    ]

    todos_sharpes = [
        punto["sharpe"]
        for punto in registros
    ]

    if markowitz is not None:
        todas_volatilidades.extend(
            markowitz["vols"]
        )
        todos_retornos.extend(
            markowitz["rets"]
        )

    x_min = min(todas_volatilidades)
    x_max = max(todas_volatilidades)

    y_min = min(todos_retornos)
    y_max = max(todos_retornos)

    padding_x = max(
        (x_max - x_min) * 0.08,
        0.0001,
    )

    padding_y = max(
        (y_max - y_min) * 0.08,
        0.0001,
    )

    sharpe_min = min(todos_sharpes)
    sharpe_max = max(todos_sharpes)

    if sharpe_min == sharpe_max:
        sharpe_max = sharpe_min + 0.0001

    figura = go.Figure()

    primera_generacion = (
        historial_generaciones[0]
    )

    poblacion_inicial = [
        punto
        for punto in primera_generacion
        if not punto["es_frente"]
    ]

    frente_inicial = sorted(
        [
            punto
            for punto in primera_generacion
            if punto["es_frente"]
        ],
        key=lambda punto: punto["vol"],
    )

    if markowitz is not None:
        figura.add_trace(
            go.Scatter(
                x=markowitz["vols"],
                y=markowitz["rets"],
                mode="lines",
                name="Frontera Markowitz",
                line={
                    "color": "#B3452F",
                    "dash": "dash",
                },
                hovertemplate=(
                    "Volatilidad=%{x:.4f}<br>"
                    "Retorno=%{y:.4f}"
                    "<extra>Markowitz</extra>"
                ),
            )
        )

    figura.add_trace(
        go.Scatter(
            x=[
                punto["vol"]
                for punto in poblacion_inicial
            ],
            y=[
                punto["ret"]
                for punto in poblacion_inicial
            ],
            mode="markers",
            name="Población",
            marker={
                "size": 8,
                "color": "rgba(140,140,140,0.45)",
            },
            hovertemplate=(
                "Volatilidad=%{x:.4f}<br>"
                "Retorno=%{y:.4f}"
                "<extra>Población</extra>"
            ),
        )
    )

    figura.add_trace(
        go.Scatter(
            x=[
                punto["vol"]
                for punto in frente_inicial
            ],
            y=[
                punto["ret"]
                for punto in frente_inicial
            ],
            mode="markers+lines",
            name="Frente de Pareto",
            marker={
                "size": 10,
                "color": [
                    punto["sharpe"]
                    for punto in frente_inicial
                ],
                "colorscale": "YlGnBu",
                "cmin": sharpe_min,
                "cmax": sharpe_max,
                "showscale": True,
                "colorbar": {
                    "title": "Sharpe",
                },
                "line": {
                    "width": 1,
                    "color": "white",
                },
            },
            line={
                "color": "rgba(70,130,180,0.40)",
                "width": 2,
            },
            hovertemplate=(
                "Volatilidad=%{x:.4f}<br>"
                "Retorno=%{y:.4f}<br>"
                "Sharpe=%{marker.color:.4f}"
                "<extra>Frente</extra>"
            ),
        )
    )

    frames = []

    for indice, snapshot in enumerate(
        historial_generaciones
    ):
        poblacion = [
            punto
            for punto in snapshot
            if not punto["es_frente"]
        ]

        frente = sorted(
            [
                punto
                for punto in snapshot
                if punto["es_frente"]
            ],
            key=lambda punto: punto["vol"],
        )

        datos_frame = []

        if markowitz is not None:
            datos_frame.append(
                go.Scatter(
                    x=markowitz["vols"],
                    y=markowitz["rets"],
                    mode="lines",
                    name="Frontera Markowitz",
                    line={
                        "color": "#B3452F",
                        "dash": "dash",
                    },
                    hovertemplate=(
                        "Volatilidad=%{x:.4f}<br>"
                        "Retorno=%{y:.4f}"
                        "<extra>Markowitz</extra>"
                    ),
                )
            )

        datos_frame.extend(
            [
                go.Scatter(
                    x=[
                        punto["vol"]
                        for punto in poblacion
                    ],
                    y=[
                        punto["ret"]
                        for punto in poblacion
                    ],
                    mode="markers",
                    name="Población",
                    marker={
                        "size": 8,
                        "color": (
                            "rgba(140,140,140,0.45)"
                        ),
                    },
                    hovertemplate=(
                        "Volatilidad=%{x:.4f}<br>"
                        "Retorno=%{y:.4f}"
                        "<extra>Población</extra>"
                    ),
                ),
                go.Scatter(
                    x=[
                        punto["vol"]
                        for punto in frente
                    ],
                    y=[
                        punto["ret"]
                        for punto in frente
                    ],
                    mode="markers+lines",
                    name="Frente de Pareto",
                    marker={
                        "size": 10,
                        "color": [
                            punto["sharpe"]
                            for punto in frente
                        ],
                        "colorscale": "YlGnBu",
                        "cmin": sharpe_min,
                        "cmax": sharpe_max,
                        "showscale": False,
                        "line": {
                            "width": 1,
                            "color": "white",
                        },
                    },
                    line={
                        "color": (
                            "rgba(70,130,180,0.40)"
                        ),
                        "width": 2,
                    },
                    hovertemplate=(
                        "Volatilidad=%{x:.4f}<br>"
                        "Retorno=%{y:.4f}<br>"
                        "Sharpe=%{marker.color:.4f}"
                        "<extra>Frente</extra>"
                    ),
                ),
            ]
        )

        frames.append(
            go.Frame(
                data=datos_frame,
                name=str(indice),
            )
        )

    figura.frames = frames

    pasos_slider = []

    for indice in range(
        len(historial_generaciones)
    ):
        etiqueta = (
            "Inicial"
            if indice == 0
            else str(indice)
        )

        pasos_slider.append(
            {
                "label": etiqueta,
                "method": "animate",
                "args": [
                    [str(indice)],
                    {
                        "mode": "immediate",
                        "frame": {
                            "duration": 250,
                            "redraw": True,
                        },
                        "transition": {
                            "duration": 0,
                        },
                    },
                ],
            }
        )

    total_generaciones = (
        len(historial_generaciones) - 1
    )

    figura.update_layout(
        title=(
            "Frente de Pareto NSGA-II — "
            f"Evolución por generación "
            f"(0 a {total_generaciones})"
        ),
        xaxis_title="Volatilidad",
        yaxis_title="Retorno",
        xaxis={
            "range": [
                x_min - padding_x,
                x_max + padding_x,
            ]
        },
        yaxis={
            "range": [
                y_min - padding_y,
                y_max + padding_y,
            ]
        },
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0,
                "y": 1.17,
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
                                    "duration": 250,
                                    "redraw": True,
                                },
                                "transition": {
                                    "duration": 0,
                                },
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
                                "transition": {
                                    "duration": 0,
                                },
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
                    "prefix": "Generación: ",
                },
                "pad": {
                    "t": 50,
                },
                "steps": pasos_slider,
            }
        ],
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
    )

    return figura


def _figura_hipervolumen_animada(
    historia_hipervolumen,
):
    """
    Construye una línea animada con el hipervolumen acumulado
    generación por generación.
    """
    generaciones = list(
        range(len(historia_hipervolumen))
    )

    y_min = min(historia_hipervolumen)
    y_max = max(historia_hipervolumen)

    padding_y = max(
        (y_max - y_min) * 0.08,
        0.000001,
    )

    figura = go.Figure()

    figura.add_trace(
        go.Scatter(
            x=[generaciones[0]],
            y=[historia_hipervolumen[0]],
            mode="lines+markers",
            name="Hipervolumen",
            line={
                "color": "#1F77B4",
                "width": 3,
            },
            marker={
                "size": 8,
            },
            hovertemplate=(
                "Generación=%{x}<br>"
                "Hipervolumen=%{y:.6f}"
                "<extra></extra>"
            ),
        )
    )

    frames = []

    for indice in range(
        len(historia_hipervolumen)
    ):
        frames.append(
            go.Frame(
                data=[
                    go.Scatter(
                        x=generaciones[: indice + 1],
                        y=historia_hipervolumen[
                            : indice + 1
                        ],
                        mode="lines+markers",
                        name="Hipervolumen",
                        line={
                            "color": "#1F77B4",
                            "width": 3,
                        },
                        marker={
                            "size": 8,
                        },
                        hovertemplate=(
                            "Generación=%{x}<br>"
                            "Hipervolumen=%{y:.6f}"
                            "<extra></extra>"
                        ),
                    )
                ],
                name=str(indice),
            )
        )

    figura.frames = frames

    pasos_slider = []

    for indice in range(
        len(historia_hipervolumen)
    ):
        etiqueta = (
            "Inicial"
            if indice == 0
            else str(indice)
        )

        pasos_slider.append(
            {
                "label": etiqueta,
                "method": "animate",
                "args": [
                    [str(indice)],
                    {
                        "mode": "immediate",
                        "frame": {
                            "duration": 250,
                            "redraw": True,
                        },
                        "transition": {
                            "duration": 0,
                        },
                    },
                ],
            }
        )

    figura.update_layout(
        title=(
            "Convergencia del hipervolumen "
            "por generación"
        ),
        xaxis_title="Generación",
        yaxis_title="Indicador de hipervolumen",
        xaxis={
            "range": [
                0,
                max(generaciones),
            ]
        },
        yaxis={
            "range": [
                y_min - padding_y,
                y_max + padding_y,
            ]
        },
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0,
                "y": 1.17,
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
                                    "duration": 250,
                                    "redraw": True,
                                },
                                "transition": {
                                    "duration": 0,
                                },
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
                                "transition": {
                                    "duration": 0,
                                },
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
                    "prefix": "Generación: ",
                },
                "pad": {
                    "t": 50,
                },
                "steps": pasos_slider,
            }
        ],
    )

    return figura


ejecutar = st.button(
    "▶️ Ejecutar NSGA-II",
    type="primary",
)


estado_nsga_disponible = all(
    clave in st.session_state
    for clave in (
        "frente_nsga2",
        "hv_historia",
        "historial_generaciones_nsga2",
    )
)


if ejecutar or estado_nsga_disponible:
    if ejecutar:
        barra_progreso = st.progress(
            0,
            text="Iniciando evolución NSGA-II...",
        )

        def actualizar_progreso(
            generacion,
            total,
        ):
            barra_progreso.progress(
                generacion / total,
                text=(
                    f"Generación "
                    f"{generacion}/{total}"
                ),
            )

        (
            frente,
            historia_hipervolumen,
            historial_generaciones,
        ) = ejecutar_nsga2(
            mu,
            cov,
            mu_pop,
            ngen,
            progress_callback=actualizar_progreso,
        )

        barra_progreso.empty()

        st.session_state["frente_nsga2"] = (
            frente
        )

        st.session_state["hv_historia"] = (
            historia_hipervolumen
        )

        st.session_state[
            "historial_generaciones_nsga2"
        ] = historial_generaciones

    else:
        frente = st.session_state[
            "frente_nsga2"
        ]

        historia_hipervolumen = (
            st.session_state["hv_historia"]
        )

        historial_generaciones = (
            st.session_state[
                "historial_generaciones_nsga2"
            ]
        )

    retornos = [
        resultado["ret"]
        for resultado in frente
    ]

    volatilidades = [
        resultado["vol"]
        for resultado in frente
    ]

    sharpes = [
        resultado["sharpe"]
        for resultado in frente
    ]

    indice_conservador = int(
        np.argmin(volatilidades)
    )

    indice_agresivo = int(
        np.argmax(retornos)
    )

    indice_balanceado = int(
        np.argmax(sharpes)
    )

    st.success(
        "Frente de Pareto calculado: "
        f"{len(frente)} soluciones no dominadas."
    )

    st.info(
        "La animación muestra cómo evoluciona la población "
        "y cómo se forma el frente no dominado en cada generación."
    )

    st.subheader(
        "Frente de Pareto NSGA-II"
    )

    datos_markowitz = None

    if "resultados_m1" in st.session_state:
        resultados_m1 = st.session_state[
            "resultados_m1"
        ]

        datos_markowitz = {
            "vols": resultados_m1[
                "frontera_vols"
            ],
            "rets": resultados_m1[
                "frontera_objetivos"
            ],
        }

    else:
        st.info(
            "Ejecuta primero el Módulo 1 para "
            "superponer la frontera de Markowitz."
        )

    figura_pareto = _figura_pareto_animada(
        historial_generaciones,
        markowitz=datos_markowitz,
    )

    st.plotly_chart(
        figura_pareto,
        use_container_width=True,
    )

    st.subheader(
        "Convergencia del hipervolumen"
    )

    figura_hipervolumen = (
        _figura_hipervolumen_animada(
            historia_hipervolumen
        )
    )

    st.plotly_chart(
        figura_hipervolumen,
        use_container_width=True,
    )

    st.subheader(
        "Portafolios representativos "
        "del frente de Pareto"
    )

    columna_1, columna_2, columna_3 = (
        st.columns(3)
    )

    with columna_1:
        figura_conservador = px.pie(
            names=tick_list,
            values=frente[
                indice_conservador
            ]["weights"],
            hole=0.3,
            title=(
                "Conservador "
                f"(σ={volatilidades[indice_conservador] * 100:.1f}%)"
            ),
        )

        st.plotly_chart(
            figura_conservador,
            use_container_width=True,
        )

    with columna_2:
        figura_balanceado = px.pie(
            names=tick_list,
            values=frente[
                indice_balanceado
            ]["weights"],
            hole=0.3,
            title=(
                "Balanceado "
                f"(Sharpe={sharpes[indice_balanceado]:.2f})"
            ),
        )

        st.plotly_chart(
            figura_balanceado,
            use_container_width=True,
        )

    with columna_3:
        figura_agresivo = px.pie(
            names=tick_list,
            values=frente[
                indice_agresivo
            ]["weights"],
            hole=0.3,
            title=(
                "Agresivo "
                f"(Retorno={retornos[indice_agresivo] * 100:.1f}%)"
            ),
        )

        st.plotly_chart(
            figura_agresivo,
            use_container_width=True,
        )

    st.subheader(
        "Simulación de riqueza — Portafolios GA"
    )

    riqueza_conservador = simular_riqueza(
        retornos_simples,
        frente[indice_conservador]["weights"],
        capital,
        freq_label=frecuencia,
    )

    riqueza_balanceado = simular_riqueza(
        retornos_simples,
        frente[indice_balanceado]["weights"],
        capital,
        freq_label=frecuencia,
    )

    riqueza_agresivo = simular_riqueza(
        retornos_simples,
        frente[indice_agresivo]["weights"],
        capital,
        freq_label=frecuencia,
    )

    figura_riqueza = go.Figure()

    figura_riqueza.add_trace(
        go.Scatter(
            x=riqueza_conservador.index,
            y=riqueza_conservador.values,
            name="Conservador",
        )
    )

    figura_riqueza.add_trace(
        go.Scatter(
            x=riqueza_balanceado.index,
            y=riqueza_balanceado.values,
            name="Balanceado",
        )
    )

    figura_riqueza.add_trace(
        go.Scatter(
            x=riqueza_agresivo.index,
            y=riqueza_agresivo.values,
            name="Agresivo",
        )
    )

    figura_riqueza.update_layout(
        title=(
            "Evolución de riqueza "
            "(GA rebalanceado)"
        ),
        xaxis_title="Fecha",
        yaxis_title=(
            "Valor del portafolio (USD)"
        ),
    )

    st.plotly_chart(
        figura_riqueza,
        use_container_width=True,
    )

    st.session_state["resultados_m2"] = {
        "tickers": tick_list,
        "frente_pareto": [
            {
                "weights": (
                    resultado["weights"].tolist()
                ),
                "ret": resultado["ret"],
                "vol": resultado["vol"],
                "sharpe": resultado["sharpe"],
            }
            for resultado in frente
        ],
        "hv_historia": historia_hipervolumen,
        "idx_conservador": indice_conservador,
        "idx_balanceado": indice_balanceado,
        "idx_agresivo": indice_agresivo,
        "wealth_conservador": serie_a_dict(
            riqueza_conservador
        ),
        "wealth_balanceado": serie_a_dict(
            riqueza_balanceado
        ),
        "wealth_agresivo": serie_a_dict(
            riqueza_agresivo
        ),
        "mu_pop": mu_pop,
        "ngen": ngen,
    }

    st.markdown("---")

    datos_pareto = []

    for resultado in frente:
        fila = {
            "Ticker_" + ticker:
                resultado["weights"][indice]
            for indice, ticker in enumerate(
                tick_list
            )
        }

        fila.update(
            {
                "Retorno": resultado["ret"],
                "Volatilidad": resultado["vol"],
                "Sharpe": resultado["sharpe"],
            }
        )

        datos_pareto.append(fila)

    df_pareto = pd.DataFrame(
        datos_pareto
    )

    st.download_button(
        "⬇️ Descargar frente de Pareto completo (Excel)",
        data=df_a_excel_bytes(
            {
                "Frente_Pareto": df_pareto,
            }
        ),
        file_name=(
            "frente_pareto_nsga2.xlsx"
        ),
        mime=(
            "application/vnd.openxmlformats-"
            "officedocument.spreadsheetml.sheet"
        ),
    )

else:
    st.info(
        "Presiona **'Ejecutar NSGA-II'** "
        "para correr el algoritmo genético."
    )
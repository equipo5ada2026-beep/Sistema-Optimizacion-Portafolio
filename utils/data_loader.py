"""
utils/data_loader.py
Descarga y preparación de datos de mercado (Yahoo Finance).
Adaptado del bloque compartido por los 4 notebooks del proyecto (Módulos 1, 2 y 3).
"""

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


@st.cache_data(show_spinner=False, ttl=3600)
def cargar_datos(tickers: list, start: str, end: str):
    """
    Descarga precios de cierre ajustados desde Yahoo Finance y calcula:
    - precios: DataFrame de precios ajustados
    - log_returns: retornos logarítmicos diarios
    - mu: vector de retornos esperados anualizados (252 sesiones)
    - cov: matriz de covarianza anualizada
    - tick_list: lista de tickers efectivamente cargados

    Cacheada por Streamlit (misma combinación tickers/fechas no se re-descarga).
    """
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    if raw.empty:
        raise ValueError(
            "Yahoo Finance no devolvió datos para los tickers/fechas indicados. "
            "Verifica que los símbolos existan y que el rango de fechas sea válido."
        )

    if isinstance(raw.columns, pd.MultiIndex):
        precios = raw["Close"]
    else:
        precios = raw[["Close"]]
        precios.columns = tickers

    precios = precios.dropna(how="all").ffill().dropna()

    if precios.empty or precios.shape[1] == 0:
        raise ValueError("No quedaron datos válidos tras limpiar valores nulos.")

    tick_list = list(precios.columns)
    log_returns = np.log(precios / precios.shift(1)).dropna()
    mu = log_returns.mean() * 252
    cov = log_returns.cov() * 252

    return precios, log_returns, mu, cov, tick_list


def parsear_tickers(texto: str) -> list:
    """Convierte el texto del st.text_input (separado por comas) en una lista limpia de tickers."""
    return [t.strip().upper() for t in texto.split(",") if t.strip()]


def fechas_rebalanceo(precios_index, freq_label: str) -> set:
    """Determina el conjunto de fechas de rebalanceo según la frecuencia elegida en el sidebar."""
    freq_map = {"Semanal": "W", "Mensual": "ME", "Trimestral": "QE"}
    freq = freq_map.get(freq_label, "M")
    serie = pd.Series(precios_index, index=precios_index)
    ultimas = serie.resample(freq).last().dropna()
    return set(ultimas.values)


def simular_riqueza(retornos_simples: pd.DataFrame, pesos_objetivo, capital: float,
                     freq_label=None) -> pd.Series:
    """
    Simula la evolución de riqueza de una cartera con pesos objetivo dados.
    freq_label=None -> Buy & Hold puro (sin rebalanceo).
    freq_label con valor -> rebalanceo periódico a los pesos objetivo.
    """
    pesos_objetivo = np.array(pesos_objetivo, dtype=float)
    fechas = retornos_simples.index
    rebal_dates = fechas_rebalanceo(fechas, freq_label) if freq_label else set()

    valores_activos = capital * pesos_objetivo
    wealth = []
    for fecha, fila in retornos_simples.iterrows():
        valores_activos = valores_activos * (1 + fila.values)
        total = valores_activos.sum()
        wealth.append(total)
        if freq_label and fecha in rebal_dates:
            valores_activos = total * pesos_objetivo

    return pd.Series(wealth, index=fechas)


def serie_a_dict(serie: pd.Series, usar_fechas: bool = True) -> dict:
    """Convierte una pd.Series de riqueza en un dict serializable (session_state / descarga)."""
    if usar_fechas:
        return {"fechas": [d.strftime("%Y-%m-%d") for d in serie.index], "valores": serie.values.tolist()}
    return {"periodos": list(range(len(serie))), "valores": serie.values.tolist()}

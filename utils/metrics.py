"""
utils/metrics.py
Métricas de desempeño de portafolio (Sharpe, Sortino, Max Drawdown).
Adaptado de: Modulo4_Comparacion.ipynb.
"""

import numpy as np
import pandas as pd


def metricas_desempeno(wealth_series, periodos_por_anio: int = 252) -> dict | None:
    """
    Calcula métricas de desempeño a partir de una serie de riqueza:
    Retorno Total, Retorno Anualizado, Volatilidad Anualizada, Sharpe, Sortino,
    Max Drawdown y Riqueza Final.

    periodos_por_anio: 252 para series diarias, 12 para series mensuales (ej. Módulo DP).
    Se asume tasa libre de riesgo (rf) = 0.
    """
    w = pd.Series(wealth_series).astype(float)
    if len(w) < 2:
        return None

    rets = w.pct_change().dropna()
    retorno_total = w.iloc[-1] / w.iloc[0] - 1
    n_periodos = len(w)
    retorno_anual = (1 + retorno_total) ** (periodos_por_anio / max(n_periodos, 1)) - 1
    vol_anual = rets.std() * np.sqrt(periodos_por_anio)
    sharpe = retorno_anual / vol_anual if vol_anual and vol_anual > 0 else np.nan

    downside = rets[rets < 0].std()
    downside_anual = downside * np.sqrt(periodos_por_anio) if downside and not np.isnan(downside) else np.nan
    sortino = retorno_anual / downside_anual if downside_anual and downside_anual > 0 else np.nan

    drawdown = w / w.cummax() - 1
    max_dd = drawdown.min()

    return {
        "Retorno Total (%)": round(retorno_total * 100, 2),
        "Retorno Anualizado (%)": round(retorno_anual * 100, 2),
        "Volatilidad Anualizada (%)": round(vol_anual * 100, 2),
        "Sharpe Ratio": round(sharpe, 2) if pd.notna(sharpe) else np.nan,
        "Sortino Ratio": round(sortino, 2) if pd.notna(sortino) else np.nan,
        "Max Drawdown (%)": round(max_dd * 100, 2),
        "Riqueza Final (USD)": round(w.iloc[-1], 2),
    }

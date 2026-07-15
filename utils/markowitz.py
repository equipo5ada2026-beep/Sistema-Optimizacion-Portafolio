"""
utils/markowitz.py
Modelo de media-varianza de Markowitz: máximo Sharpe, mínima varianza y frontera eficiente.
Adaptado de: Modulo1_Datos_Markowitz.ipynb (usado también por el Módulo 3 - DP).
"""

import numpy as np
from scipy.optimize import minimize


def rendimiento_portafolio(pesos, mu, cov):
    """Retorno y volatilidad esperados (anualizados) de una cartera con pesos dados."""
    ret = float(np.dot(pesos, mu))
    vol = float(np.sqrt(pesos @ cov @ pesos))
    return ret, vol


def _neg_sharpe(pesos, mu, cov, rf=0.0):
    ret, vol = rendimiento_portafolio(pesos, mu, cov)
    return -(ret - rf) / vol if vol > 0 else 0.0


def _volatilidad(pesos, mu, cov):
    return rendimiento_portafolio(pesos, mu, cov)[1]


def optimizar_max_sharpe(mu, cov, rf: float = 0.0):
    """Resuelve el portafolio de máximo Sharpe Ratio (SLSQP, sin ventas en corto, suma pesos = 1)."""
    n = len(mu)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    init = np.repeat(1.0 / n, n)
    res = minimize(_neg_sharpe, init, args=(mu.values, cov.values, rf),
                    method="SLSQP", bounds=bounds, constraints=cons)
    return res.x if res.success else init


def optimizar_min_varianza(mu, cov):
    """Resuelve el portafolio de mínima varianza global (SLSQP, mismas restricciones)."""
    n = len(mu)
    bounds = tuple((0.0, 1.0) for _ in range(n))
    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)
    init = np.repeat(1.0 / n, n)
    res = minimize(_volatilidad, init, args=(mu.values, cov.values),
                    method="SLSQP", bounds=bounds, constraints=cons)
    return res.x if res.success else init


def frontera_eficiente(mu, cov, n_puntos: int = 200):
    """
    Genera la frontera eficiente resolviendo el problema de mínima varianza
    para n_puntos niveles de retorno objetivo entre min(mu) y max(mu).
    """
    n = len(mu)
    objetivos = np.linspace(mu.min(), mu.max(), n_puntos)
    vols, pesos_list = [], []
    bounds = tuple((0.0, 1.0) for _ in range(n))
    init = np.repeat(1.0 / n, n)

    for target in objetivos:
        cons = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, t=target: float(np.dot(w, mu.values)) - t},
        )
        res = minimize(_volatilidad, init, args=(mu.values, cov.values),
                        method="SLSQP", bounds=bounds, constraints=cons)
        if res.success:
            vols.append(res.fun)
            pesos_list.append(res.x)
        else:
            vols.append(np.nan)
            pesos_list.append(np.full(n, np.nan))

    return objetivos, np.array(vols), pesos_list

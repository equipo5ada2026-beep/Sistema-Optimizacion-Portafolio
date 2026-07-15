"""
utils/dp_rebalanceo.py
Rebalanceo óptimo vía Programación Dinámica (backward induction de Bellman).
Adaptado de: Modulo3_DP_Rebalanceo.ipynb.

Estado s_t en [0,1]: fracción del capital expuesta al portafolio tangente
(máximo Sharpe); el resto queda en efectivo (retorno 0).
"""

import numpy as np
import pandas as pd


def ejecutar_dp_rebalanceo(mu_p: float, sigma_p: float, costo_trans_pct: float,
                           horizonte: int, grid_step: float, aversion_riesgo: float = 3.0):
    """
    Backward induction de la ecuación de Bellman:
        J*_t(s_{t-1}) = max_{s_t in grid} [ r(s_t) - c(s_{t-1}, s_t) + J*_{t+1}(s_t) ]
        J*_T(.) = 0
    donde r(s) = s*mu_p - 0.5*lambda*s^2*sigma_p^2 (utilidad tipo Merton)
    y c(s_{t-1}, s_t) = costo * |s_t - s_{t-1}| (costo de ajuste lineal).
    """
    grid = np.round(np.arange(0.0, 1.0 + grid_step, grid_step), 6)
    grid = grid[grid <= 1.0 + 1e-9]
    n_estados = len(grid)

    recompensa = grid * mu_p - 0.5 * aversion_riesgo * (grid ** 2) * (sigma_p ** 2)
    costo_pct = costo_trans_pct / 100.0
    matriz_costos = costo_pct * np.abs(grid.reshape(-1, 1) - grid.reshape(1, -1))

    J = np.zeros((horizonte + 1, n_estados))
    politica = np.zeros((horizonte, n_estados), dtype=int)

    for t in range(horizonte - 1, -1, -1):
        for i in range(n_estados):
            valores = recompensa - matriz_costos[i, :] + J[t + 1, :]
            j_best = int(np.argmax(valores))
            J[t, i] = valores[j_best]
            politica[t, i] = j_best

    return grid, J, politica, matriz_costos


def extraer_decisiones(grid, politica, horizonte: int, estado_inicial: float = 0.5):
    """Extrae la trayectoria de decisiones óptimas partiendo de un estado inicial neutro."""
    estado_actual = int(np.argmin(np.abs(grid - estado_inicial)))
    decisiones = []
    for t in range(horizonte):
        siguiente = politica[t, estado_actual]
        delta = grid[siguiente] - grid[estado_actual]
        if abs(delta) < 1e-9:
            decisiones.append("No Cambiar")
        elif delta > 0:
            decisiones.append("Rebalancear Compra")
        else:
            decisiones.append("Rebalancear Venta")
        estado_actual = siguiente

    return pd.DataFrame({"Periodo": [f"T_{i}" for i in range(horizonte)], "Decision Optima": decisiones})


def simular_dp_forward(grid, politica, retornos_periodicos: pd.Series, costo_trans_pct: float,
                        capital: float, horizonte: int):
    """
    Simula hacia adelante 3 estrategias con los retornos periódicos reales:
    - DP Óptimo (sigue la política de Bellman)
    - Buy & Hold (mantiene la exposición inicial fija)
    - Siempre Rebalanceado a exposición total (w=1)
    """
    horizonte = min(horizonte, len(retornos_periodicos))
    retornos = retornos_periodicos.values[:horizonte]
    fechas = retornos_periodicos.index[:horizonte]
    costo_pct = costo_trans_pct / 100.0
    idx_ini = int(np.argmin(np.abs(grid - 0.5)))

    wealth_dp, w_cur, idx_cur = [capital], grid[idx_ini], idx_ini
    for t in range(horizonte):
        r = retornos[t]
        idx_next = politica[t, idx_cur]
        w_next = grid[idx_next]
        costo = costo_pct * abs(w_next - w_cur)
        wealth_dp.append(wealth_dp[-1] * (1 + w_cur * r) * (1 - costo))
        w_cur, idx_cur = w_next, idx_next

    w_fijo = grid[idx_ini]
    wealth_bh = [capital]
    for t in range(horizonte):
        wealth_bh.append(wealth_bh[-1] * (1 + w_fijo * retornos[t]))

    wealth_full = [capital]
    for t in range(horizonte):
        wealth_full.append(wealth_full[-1] * (1 + 1.0 * retornos[t]))

    return fechas, wealth_dp[1:], wealth_bh[1:], wealth_full[1:]

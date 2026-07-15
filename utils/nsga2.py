"""
utils/nsga2.py
Algoritmo genético NSGA-II (DEAP) para optimización bi-objetivo del portafolio
(maximizar retorno, minimizar riesgo). Adaptado de: Modulo2_NSGA2_Multiobjetivo.ipynb.
"""

import random
import numpy as np
from deap import base, creator, tools

SEED = 42


def _normalizar_individuo(ind, n):
    """Normaliza un individuo (genes en [0,1]) a pesos válidos que suman 1."""
    arr = np.clip(np.array(ind, dtype=float), 0.0, None)
    s = arr.sum()
    return (arr / s) if s > 0 else np.repeat(1.0 / n, n)


def hipervolumen_2d(frente, ref_ret, ref_vol):
    """Indicador de hypervolumen en 2D (retorno vs. volatilidad) respecto a un punto de referencia."""
    pts = sorted(((ind.fitness.values[0], ind.fitness.values[1]) for ind in frente), key=lambda p: p[1])
    hv, prev_vol = 0.0, ref_vol
    for ret, vol in pts:
        if vol < prev_vol:
            hv += max(0.0, prev_vol - vol) * max(0.0, ret - ref_ret)
            prev_vol = vol
    return hv


def configurar_toolbox(mu_arr, cov_arr, n_activos):
    """
    Configura el toolbox de DEAP: individuo (vector de pesos en [0,1]), función de
    evaluación bi-objetivo, y operadores genéticos (cruce, mutación, selección NSGA-II).
    """
    random.seed(SEED)

    if not hasattr(creator, "FitnessMulti"):
        creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMulti)

    def evaluar(ind):
        w = _normalizar_individuo(ind, n_activos)
        ret = float(np.dot(w, mu_arr))
        vol = float(np.sqrt(w @ cov_arr @ w))
        return ret, vol

    toolbox = base.Toolbox()
    toolbox.register("attr_float", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_activos)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluar)
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=20.0)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=1.0 / n_activos)
    toolbox.register("select", tools.selNSGA2)
    return toolbox


def ejecutar_nsga2(mu, cov, mu_pop: int, ngen: int, cxpb: float = 0.7, mutpb: float = 0.3,
                    progress_callback=None):
    """
    Ejecuta el ciclo evolutivo NSGA-II completo:
    selTournamentDCD -> cruce -> mutación -> selNSGA2 (nueva generación).
    Calcula el hypervolumen en cada generación para medir convergencia.

    progress_callback(gen, ngen): función opcional invocada en cada generación,
    usada por la página de Streamlit para actualizar un st.progress().
    """
    n_activos = len(mu)
    mu_arr, cov_arr = mu.values, cov.values
    toolbox = configurar_toolbox(mu_arr, cov_arr, n_activos)

    pop_size = max(8, int(round(mu_pop / 4.0)) * 4)  # múltiplo de 4 (requisito de selTournamentDCD)
    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    pop = toolbox.select(pop, len(pop))

    ref_ret = min(mu_arr.min(), 0.0) - 0.05
    ref_vol = float(np.sqrt(np.diag(cov_arr)).max()) * 1.2

    hv_historia = []
    for gen in range(ngen):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() <= cxpb:
                toolbox.mate(c1, c2)
            if random.random() <= mutpb:
                toolbox.mutate(c1)
                toolbox.mutate(c2)
            del c1.fitness.values, c2.fitness.values

        invalidos = [ind for ind in offspring if not ind.fitness.valid]
        for ind in invalidos:
            ind.fitness.values = toolbox.evaluate(ind)

        pop = toolbox.select(pop + offspring, pop_size)
        frente0 = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        hv_historia.append(hipervolumen_2d(frente0, ref_ret, ref_vol))

        if progress_callback:
            progress_callback(gen + 1, ngen)

    frente_final = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    resultados = []
    for ind in frente_final:
        w = _normalizar_individuo(ind, n_activos)
        ret, vol = ind.fitness.values
        sharpe = ret / vol if vol > 0 else 0.0
        resultados.append({"weights": w, "ret": ret, "vol": vol, "sharpe": sharpe})
    resultados.sort(key=lambda r: r["vol"])

    return resultados, hv_historia

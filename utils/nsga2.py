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
    """Normaliza un individuo a pesos válidos que suman 1."""
    arr = np.clip(np.array(ind, dtype=float), 0.0, None)
    suma = arr.sum()

    if suma > 0:
        return arr / suma

    return np.repeat(1.0 / n, n)


def hipervolumen_2d(frente, ref_ret, ref_vol):
    """
    Calcula el indicador de hipervolumen en dos dimensiones:
    retorno frente a volatilidad.
    """
    puntos = sorted(
        (
            (ind.fitness.values[0], ind.fitness.values[1])
            for ind in frente
        ),
        key=lambda punto: punto[1],
    )

    hipervolumen = 0.0
    volatilidad_anterior = ref_vol

    for retorno, volatilidad in puntos:
        if volatilidad < volatilidad_anterior:
            hipervolumen += (
                max(0.0, volatilidad_anterior - volatilidad)
                * max(0.0, retorno - ref_ret)
            )
            volatilidad_anterior = volatilidad

    return hipervolumen


def configurar_toolbox(mu_arr, cov_arr, n_activos):
    """
    Configura los individuos, la evaluación y los operadores genéticos
    utilizados por NSGA-II.
    """
    random.seed(SEED)

    if not hasattr(creator, "FitnessMulti"):
        creator.create(
            "FitnessMulti",
            base.Fitness,
            weights=(1.0, -1.0),
        )

    if not hasattr(creator, "Individual"):
        creator.create(
            "Individual",
            list,
            fitness=creator.FitnessMulti,
        )

    def evaluar(individuo):
        pesos = _normalizar_individuo(individuo, n_activos)

        retorno = float(np.dot(pesos, mu_arr))
        volatilidad = float(
            np.sqrt(pesos @ cov_arr @ pesos)
        )

        return retorno, volatilidad

    toolbox = base.Toolbox()

    toolbox.register(
        "attr_float",
        random.random,
    )

    toolbox.register(
        "individual",
        tools.initRepeat,
        creator.Individual,
        toolbox.attr_float,
        n=n_activos,
    )

    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.individual,
    )

    toolbox.register(
        "evaluate",
        evaluar,
    )

    toolbox.register(
        "mate",
        tools.cxSimulatedBinaryBounded,
        low=0.0,
        up=1.0,
        eta=20.0,
    )

    toolbox.register(
        "mutate",
        tools.mutPolynomialBounded,
        low=0.0,
        up=1.0,
        eta=20.0,
        indpb=1.0 / n_activos,
    )

    toolbox.register(
        "select",
        tools.selNSGA2,
    )

    return toolbox


def _snapshot_generacion(poblacion, n_activos):
    """
    Guarda una fotografía de la población actual.

    Cada punto conserva:
    - retorno;
    - volatilidad;
    - ratio de Sharpe;
    - pesos;
    - pertenencia al frente no dominado.
    """
    frente = tools.sortNondominated(
        poblacion,
        len(poblacion),
        first_front_only=True,
    )[0]

    identificadores_frente = {
        id(individuo)
        for individuo in frente
    }

    snapshot = []

    for indice, individuo in enumerate(poblacion):
        pesos = _normalizar_individuo(
            individuo,
            n_activos,
        )

        retorno, volatilidad = individuo.fitness.values

        sharpe = (
            retorno / volatilidad
            if volatilidad > 0
            else 0.0
        )

        snapshot.append(
            {
                "idx": indice,
                "weights": pesos,
                "ret": float(retorno),
                "vol": float(volatilidad),
                "sharpe": float(sharpe),
                "es_frente": (
                    id(individuo)
                    in identificadores_frente
                ),
            }
        )

    return snapshot, frente


def ejecutar_nsga2(
    mu,
    cov,
    mu_pop: int,
    ngen: int,
    cxpb: float = 0.7,
    mutpb: float = 0.3,
    progress_callback=None,
):
    """
    Ejecuta NSGA-II y guarda la evolución completa de la población.

    Retorna:
    - frente final;
    - historia del hipervolumen;
    - historia de las poblaciones por generación.
    """
    n_activos = len(mu)

    mu_arr = mu.values
    cov_arr = cov.values

    toolbox = configurar_toolbox(
        mu_arr,
        cov_arr,
        n_activos,
    )

    # selTournamentDCD requiere una población múltiplo de cuatro.
    pop_size = max(
        8,
        int(round(mu_pop / 4.0)) * 4,
    )

    poblacion = toolbox.population(
        n=pop_size
    )

    for individuo in poblacion:
        individuo.fitness.values = toolbox.evaluate(
            individuo
        )

    poblacion = toolbox.select(
        poblacion,
        len(poblacion),
    )

    ref_ret = min(
        mu_arr.min(),
        0.0,
    ) - 0.05

    ref_vol = (
        float(
            np.sqrt(
                np.diag(cov_arr)
            ).max()
        )
        * 1.2
    )

    historia_hipervolumen = []
    historial_generaciones = []

    # Generación inicial: generación 0.
    snapshot_inicial, frente_inicial = (
        _snapshot_generacion(
            poblacion,
            n_activos,
        )
    )

    historial_generaciones.append(
        snapshot_inicial
    )

    historia_hipervolumen.append(
        hipervolumen_2d(
            frente_inicial,
            ref_ret,
            ref_vol,
        )
    )

    for generacion in range(ngen):
        descendencia = tools.selTournamentDCD(
            poblacion,
            len(poblacion),
        )

        descendencia = [
            toolbox.clone(individuo)
            for individuo in descendencia
        ]

        for hijo_1, hijo_2 in zip(
            descendencia[::2],
            descendencia[1::2],
        ):
            if random.random() <= cxpb:
                toolbox.mate(
                    hijo_1,
                    hijo_2,
                )

            if random.random() <= mutpb:
                toolbox.mutate(hijo_1)
                toolbox.mutate(hijo_2)

            del hijo_1.fitness.values
            del hijo_2.fitness.values

        individuos_invalidos = [
            individuo
            for individuo in descendencia
            if not individuo.fitness.valid
        ]

        for individuo in individuos_invalidos:
            individuo.fitness.values = (
                toolbox.evaluate(individuo)
            )

        poblacion = toolbox.select(
            poblacion + descendencia,
            pop_size,
        )

        snapshot, frente_actual = (
            _snapshot_generacion(
                poblacion,
                n_activos,
            )
        )

        historial_generaciones.append(
            snapshot
        )

        historia_hipervolumen.append(
            hipervolumen_2d(
                frente_actual,
                ref_ret,
                ref_vol,
            )
        )

        if progress_callback:
            progress_callback(
                generacion + 1,
                ngen,
            )

    frente_final = tools.sortNondominated(
        poblacion,
        len(poblacion),
        first_front_only=True,
    )[0]

    resultados = []

    for individuo in frente_final:
        pesos = _normalizar_individuo(
            individuo,
            n_activos,
        )

        retorno, volatilidad = (
            individuo.fitness.values
        )

        sharpe = (
            retorno / volatilidad
            if volatilidad > 0
            else 0.0
        )

        resultados.append(
            {
                "weights": pesos,
                "ret": retorno,
                "vol": volatilidad,
                "sharpe": sharpe,
            }
        )

    resultados.sort(
        key=lambda resultado: resultado["vol"]
    )

    return (
        resultados,
        historia_hipervolumen,
        historial_generaciones,
    )
# Sistema de Optimización de Portafolio — GA + DP

Sistema web interactivo desarrollado en **Streamlit** que integra y compara cuatro métodos de optimización de portafolios de inversión: **Markowitz** (media-varianza), **NSGA-II** (algoritmo genético multiobjetivo), **Programación Dinámica** (rebalanceo óptimo) y **Buy & Hold**.

Proyecto desarrollado para el curso **Análisis y Diseño de Algoritmos (ADA)** — Facultad de Ingeniería de Sistemas e Informática (FISI), Universidad Nacional Mayor de San Marcos (UNMSM), Ciclo 2026.

**Docente:** Prof. Mg. Ing. Ernesto D. Cancho-Rodríguez, MBA (The George Washington University)

---

## 🔗 Enlaces

| Recurso | URL |
|---|---|
| Sistema desplegado | `https://grupo5-optimizacion-portafolio.streamlit.app/` |
| Repositorio GitHub | `https://github.com/equipo5ada2026-beep/Sistema-Optimizacion-Portafolio` |
| Video demo (YouTube) | `[completar URL]` |

---

## 👥 Equipo

| Apellidos, Nombres | Código |
|---|---|
| ... | ... |
| Martines Cancho, Josue | 24200060 |
| Languasco Llauca, Ariana Milagros | 24200225 |
| Rivera Bonifacio, Leonardo Tadeo | 24200063 |
| ... | ... |
| ... | ... |
| Duran Obeso, Jeremy Alexander | 24200056 |
| ... | ... |
| ... | ... |

---

## 📋 Descripción del sistema

El sistema descarga precios históricos desde **Yahoo Finance** para un conjunto de tickers (default: mineras con operaciones en Perú — `FSM, VOLCABC1.LM, ABX.TO, BVN, BHP`), y ejecuta cuatro módulos de análisis sobre un capital inicial de **USD $100,000** durante el horizonte **2015-01-01 a 2024-12-31**.

### Módulo 1 — Datos y Markowitz
Calcula retornos logarítmicos, resuelve los portafolios de máximo Sharpe y mínima varianza (`scipy.optimize`, SLSQP), genera la frontera eficiente y simula la evolución de riqueza (Buy&Hold vs. Markowitz rebalanceado).

### Módulo 2 — NSGA-II Multiobjetivo
Implementa el algoritmo genético NSGA-II (`DEAP`) para optimización bi-objetivo (retorno vs. riesgo), genera la frontera de Pareto, extrae 3 portafolios representativos y calcula el hypervolume.

### Módulo 3 — DP Rebalanceo
Implementa *backward induction* de la ecuación de Bellman para determinar la política óptima de rebalanceo periódico considerando costos de transacción.

### Módulo 4 — Comparación Cruzada
Compara los cuatro métodos (Markowitz, NSGA-II, DP, Buy&Hold) con tabla de métricas, gráficos superpuestos de evolución de riqueza y ranking automático por Sharpe Ratio y riqueza final.

---

## 🗂️ Estructura del repositorio

```
portfolio_optimizer/
├── app.py                    # Homepage y configuración global (sidebar)
├── requirements.txt
├── pages/
│   ├── 1_Datos_y_Markowitz.py
│   ├── 2_NSGA2_Multiobjetivo.py
│   ├── 3_DP_Rebalanceo.py
│   └── 4_Comparacion.py
├── utils/                    # Funciones compartidas entre módulos
├── notebooks/                # Notebooks originales ejecutados en Colab
├── diagramas/                # Diagrama de componentes UML (PlantUML)
```

---

## ⚙️ Tecnologías utilizadas

- [Python 3.10+](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [yfinance](https://pypi.org/project/yfinance/)
- [DEAP](https://deap.readthedocs.io/) (NSGA-II)
- [SciPy](https://scipy.org/) (`optimize.minimize`, SLSQP)
- [Plotly](https://plotly.com/python/)
- [Matplotlib](https://matplotlib.org/)
- [openpyxl](https://openpyxl.readthedocs.io/) (exportación a Excel)
- [Google Colab](https://colab.research.google.com/)

---

## 🚀 Cómo ejecutar localmente

```bash
git clone [URL del repo]
cd portfolio_optimizer
pip install -r requirements.txt
streamlit run app.py
```

## ☁️ Despliegue

Desplegado en **Streamlit Community Cloud** (https://share.streamlit.io/) conectado directamente a este repositorio, branch `main`, archivo principal `app.py`.

---

## 🤖 Uso de Inteligencia Artificial

Este proyecto empleó herramientas de IA generativa (Claude AI, ChatGPT, Gemini) para la conversión de notebooks a Streamlit, generación de código y depuración, siguiendo prompts hiper-detallados documentados en la especificación del curso. El detalle completo del proceso (herramienta, prompt, resultado, modificaciones manuales) se documenta en el Informe Word, sección "Proceso de uso de IA".

---

## 📄 Referencias bibliográficas

- Markowitz, H. (1952). *Portfolio Selection*. Journal of Finance, 7(1), 77-91. https://doi.org/10.2307/2975974
- Deb, K., Pratap, A., Agarwal, S. & Meyarivan, T. (2002). *A fast and elitist multiobjective genetic algorithm: NSGA-II*. IEEE Trans. Evolutionary Computation, 6(2), 182-197. https://doi.org/10.1109/4235.996017
- Vaezi Jezeie, F. et al. (2022). *Constrained portfolio optimization with discrete variables: An algorithmic method based on dynamic programming*. PLoS ONE, 17(8), e0271811. https://doi.org/10.1371/journal.pone.0271811

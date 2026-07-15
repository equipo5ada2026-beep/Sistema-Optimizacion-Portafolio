# Plantilla — Documentación Función por Función (para el Informe Word)

> Copiar esta estructura en el capítulo "Descripción de los módulos" / "Código explicado función por función" del Informe Word. Duplicar el bloque por cada función relevante de cada archivo `.py`.

---

## Archivo: `utils/data_loader.py`

### Función: `cargar_datos(tickers, fecha_inicio, fecha_fin)`

| Campo | Detalle |
|---|---|
| **Propósito** | Descarga precios de cierre ajustados desde Yahoo Finance para la lista de tickers y el rango de fechas indicado. |
| **Parámetros de entrada** | `tickers` (list[str]), `fecha_inicio` (date), `fecha_fin` (date) |
| **Valor de retorno** | `pd.DataFrame` con precios de cierre ajustados, índice de fechas, una columna por ticker |
| **Algoritmo / lógica** | Llama a `yfinance.download()`, limpia valores nulos, calcula retornos logarítmicos diarios `np.log(P_t / P_t-1)` |
| **Decorador Streamlit** | `@st.cache_data` (evita re-descargar en cada interacción del usuario) |
| **Excepciones manejadas** | Ticker inválido / sin datos → `st.error()` y detiene ejecución con `st.stop()` |

*(Repetir esta tabla para cada función: `calcular_retornos_anualizados`, `calcular_matriz_covarianza`, etc.)*

---

## Archivo: `utils/markowitz.py`

### Función: `optimizar_sharpe(mu, sigma, capital)`

| Campo | Detalle |
|---|---|
| **Propósito** | Encuentra el vector de pesos que maximiza el Sharpe Ratio del portafolio. |
| **Parámetros de entrada** | `mu` (vector de retornos esperados anualizados), `sigma` (matriz de covarianza anualizada), `capital` (float) |
| **Valor de retorno** | `dict` con pesos óptimos, retorno esperado, volatilidad y Sharpe Ratio |
| **Algoritmo / lógica** | `scipy.optimize.minimize` método **SLSQP**, función objetivo = `-Sharpe`, restricción `sum(pesos) = 1`, bounds `[0,1]` por activo (no venta en corto) |
| **Complejidad** | Depende de SLSQP (iterativo, no polinomial cerrado); se documenta el número de iteraciones típico |

### Función: `optimizar_min_varianza(mu, sigma)`
*(misma tabla)*

### Función: `generar_frontera_eficiente(mu, sigma, n_puntos=200)`
*(misma tabla — explicar el barrido de retornos objetivo y la restricción de igualdad)*

---

## Archivo: `utils/nsga2.py`

### Función: `configurar_toolbox_deap(n_activos)`

| Campo | Detalle |
|---|---|
| **Propósito** | Define el individuo (vector de pesos), la función de evaluación bi-objetivo (retorno, riesgo) y los operadores genéticos (cruce, mutación, selección NSGA-II). |
| **Parámetros de entrada** | `n_activos` (int) |
| **Valor de retorno** | Objeto `deap.base.Toolbox` configurado |
| **Operadores DEAP** | `cxSimulatedBinaryBounded` (cruce), `mutPolynomialBounded` (mutación), `selNSGA2` (selección) |

### Función: `ejecutar_nsga2(mu, sigma, mu_poblacion, n_generaciones)`

| Campo | Detalle |
|---|---|
| **Propósito** | Corre el algoritmo evolutivo NSGA-II durante `n_generaciones` con población `mu_poblacion`. |
| **Valor de retorno** | Frente de Pareto final (`pop`), historial de hypervolume por generación |
| **Complejidad** | O(generaciones × población² ) por el ordenamiento no dominado de NSGA-II |

### Función: `extraer_portafolios_representativos(frente_pareto)`
*(misma tabla — explicar criterio de selección: conservador = mínimo riesgo, agresivo = máximo retorno, máx Sharpe = mejor ratio)*

---

## Archivo: `utils/dp_rebalanceo.py`

### Función: `construir_tabla_dp(estados, T, lambda_tc)`

| Campo | Detalle |
|---|---|
| **Propósito** | Implementa la ecuación de Bellman mediante *backward induction* para hallar la política óptima de rebalanceo minimizando costos de transacción. |
| **Parámetros de entrada** | `estados` (discretización del espacio de pesos), `T` (horizonte, nº de periodos), `lambda_tc` (costo de transacción) |
| **Valor de retorno** | Tabla DP (costo óptimo por estado/periodo), política óptima (rebalancear / no rebalancear) |
| **Recurrencia** | `V_t(s) = min( costo_no_rebalancear, costo_rebalancear + λ_TC )` propagada hacia atrás desde `t=T` hasta `t=0` |
| **Complejidad** | O(T × \|estados\|²) |

### Función: `simular_estrategias(politica_dp, precios, capital)`
*(misma tabla — explicar cómo se simulan Buy&Hold, DP y Siempre-Rebalanceado en paralelo)*

---

## Archivo: `utils/metrics.py`

### Función: `sharpe_ratio(retornos, rf=0.0)`
### Función: `sortino_ratio(retornos, rf=0.0)`
### Función: `max_drawdown(serie_riqueza)`
*(misma tabla para cada una — indicar fórmula matemática exacta usada)*

---

## Archivo: `utils/export_excel.py`

### Función: `exportar_a_excel(df, nombre_hoja)`

| Campo | Detalle |
|---|---|
| **Propósito** | Convierte un DataFrame a bytes de un archivo `.xlsx` en memoria para `st.download_button()`. |
| **Librería** | `openpyxl` vía `pd.ExcelWriter` |
| **Valor de retorno** | `BytesIO` listo para descarga |

---

## Archivo: `pages/1_Datos_y_Markowitz.py` … `4_Comparacion.py`

Para cada página de Streamlit, documentar (sin necesidad de tabla, en prosa breve):
1. Qué parámetros toma de `st.session_state`.
2. Qué funciones de `utils/` invoca y en qué orden.
3. Qué widgets de salida usa (`st.pyplot`, `st.plotly_chart`, `st.metric`, `st.download_button`).
4. Captura de pantalla de la interfaz (insertar imagen real del sistema desplegado).

---

### 📌 Notas para completar esta plantilla
- Sustituir cada tabla con la **firma real** de la función tal como quedó en el código.
- Si una función usa un algoritmo distinto al descrito aquí, actualizar la fila "Algoritmo / lógica".
- Mantener consistencia entre esta sección y el capítulo de "Resultados" (las métricas mostradas deben coincidir con las fórmulas documentadas aquí).
- Recordar formato exigido: **Times New Roman 12pt, interlineado sencillo**.

"""
app.py — Homepage del Sistema de Optimización de Portafolio (GA + DP)
UNMSM - FISI - Análisis y Diseño de Algoritmos (ADA)
"""

import streamlit as st
from datetime import date

st.set_page_config(
    page_title="Optimización de Portafolio · GA + DP",
    page_icon="📈",
    layout="wide",
)

# ============================================================
# ESTILOS (paleta institucional definida en la especificación)
# ============================================================
PRIMARIO = "#1F3864"   # azul oscuro
ACENTO = "#800000"     # granate
DORADO = "#C5961A"

st.markdown(f"""
    <style>
        .stApp {{ background-color: #FFFFFF; }}
        h1, h2, h3 {{ color: {PRIMARIO}; font-family: 'Calibri', sans-serif; }}
        .stButton>button {{
            background-color: {PRIMARIO}; color: white; border-radius: 6px; border: none;
        }}
        .stButton>button:hover {{ background-color: {ACENTO}; color: white; }}
    </style>
""", unsafe_allow_html=True)

# ============================================================
# SIDEBAR — parámetros globales compartidos por los 4 módulos
# ============================================================
st.sidebar.title("⚙️ Parámetros del Sistema")

tickers_input = st.sidebar.text_input(
    "Tickers (separados por coma)",
    value=st.session_state.get("tickers_input", "FSM, VOLCABC1.LM, ABX.TO, BVN, BHP"),
)

col1, col2 = st.sidebar.columns(2)
fecha_inicio = col1.date_input("Fecha inicio", value=st.session_state.get("fecha_inicio", date(2015, 1, 1)))
fecha_fin = col2.date_input("Fecha fin", value=st.session_state.get("fecha_fin", date(2024, 12, 31)))

capital = st.sidebar.number_input(
    "Capital inicial (USD)", min_value=1000, step=1000,
    value=st.session_state.get("capital", 100000),
)

st.sidebar.markdown("---")
st.sidebar.subheader("NSGA-II")
mu_pop = st.sidebar.slider("Población (MU)", 50, 300, st.session_state.get("mu_pop", 100), step=4)
ngen = st.sidebar.slider("Generaciones (NGEN)", 30, 200, st.session_state.get("ngen", 80))

st.sidebar.markdown("---")
st.sidebar.subheader("Programación Dinámica")
lambda_tc = st.sidebar.slider(
    "Costo transacción λ_TC", 0.0001, 0.01, st.session_state.get("lambda_tc", 0.001),
    step=0.0001, format="%.4f",
)
horizonte_dp = st.sidebar.slider("Horizonte (periodos)", 4, 52, st.session_state.get("horizonte_dp", 12))

st.sidebar.markdown("---")
frecuencia = st.sidebar.selectbox(
    "Frecuencia de rebalanceo", ["Semanal", "Mensual", "Trimestral"],
    index=["Semanal", "Mensual", "Trimestral"].index(st.session_state.get("frecuencia", "Mensual")),
)

st.sidebar.markdown("---")
cargar = st.sidebar.button("📥 Cargar Datos (Yahoo Finance)", use_container_width=True)

# Persistir parámetros en session_state para que las páginas los reutilicen
st.session_state["tickers_input"] = tickers_input
st.session_state["fecha_inicio"] = fecha_inicio
st.session_state["fecha_fin"] = fecha_fin
st.session_state["capital"] = capital
st.session_state["mu_pop"] = mu_pop
st.session_state["ngen"] = ngen
st.session_state["lambda_tc"] = lambda_tc
st.session_state["horizonte_dp"] = horizonte_dp
st.session_state["frecuencia"] = frecuencia

if cargar:
    from utils.data_loader import cargar_datos, parsear_tickers

    tickers = parsear_tickers(tickers_input)
    with st.spinner("Descargando datos de Yahoo Finance..."):
        try:
            precios, log_returns, mu, cov, tick_list = cargar_datos(
                tickers, fecha_inicio.isoformat(), fecha_fin.isoformat()
            )
            st.session_state["datos_cargados"] = True
            st.session_state["precios"] = precios
            st.session_state["log_returns"] = log_returns
            st.session_state["mu"] = mu
            st.session_state["cov"] = cov
            st.session_state["tick_list"] = tick_list
            st.sidebar.success(f"✅ {len(tick_list)} activos cargados ({len(precios)} sesiones)")
        except Exception as e:
            st.sidebar.error(f"❌ Error al cargar datos: {e}")

# ============================================================
# CONTENIDO PRINCIPAL — Homepage
# ============================================================
st.title("📈 Sistema de Optimización de Portafolio")
st.subheader("Algoritmos Genéticos (NSGA-II) + Programación Dinámica")

st.markdown("""
Este sistema integra y compara **cuatro métodos** de optimización de portafolios de inversión,
usando datos históricos reales descargados desde **Yahoo Finance**.

Usa el panel lateral (⚙️ **Parámetros del Sistema**) para configurar tickers, fechas, capital
inicial y los hiperparámetros de cada algoritmo, luego presiona **"Cargar Datos"** y navega
por los módulos.
""")

st.markdown("### 🧭 Módulos del sistema")

c1, c2 = st.columns(2)
with c1:
    st.markdown("""
    **1️⃣ Datos y Markowitz**
    Frontera eficiente media-varianza, portafolios de máximo Sharpe y mínima varianza
    (`scipy.optimize`), simulación Buy & Hold vs. rebalanceado.

    **2️⃣ NSGA-II Multiobjetivo**
    Algoritmo genético (`DEAP`) para la frontera de Pareto retorno-riesgo, 3 portafolios
    representativos y evolución del hypervolumen.
    """)
with c2:
    st.markdown("""
    **3️⃣ DP Rebalanceo**
    Backward induction de la ecuación de Bellman para la política óptima de rebalanceo
    con costos de transacción.

    **4️⃣ Comparación Cruzada**
    Tabla de métricas, gráfico superpuesto de riqueza y ranking automático de los
    4 métodos por Sharpe Ratio.
    """)

st.markdown("---")

if st.session_state.get("datos_cargados"):
    st.success(
        f"Datos cargados: **{', '.join(st.session_state['tick_list'])}** · "
        f"{len(st.session_state['precios'])} sesiones · "
        f"Capital: ${st.session_state['capital']:,.0f}"
    )
    st.info("👈 Continúa con el **Módulo 1 (Datos y Markowitz)** en el menú de páginas.")
else:
    st.warning("👈 Presiona **'Cargar Datos'** en el panel lateral para comenzar.")

st.markdown("---")
st.caption(
    "⚠️ Los datos son simulaciones con fines académicos y no constituyen asesoría de inversión. · "
    "Curso ADA — FISI — UNMSM · Prof. Mg. Ing. Ernesto D. Cancho-Rodríguez, MBA"
)

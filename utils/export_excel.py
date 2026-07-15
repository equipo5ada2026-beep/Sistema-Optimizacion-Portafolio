"""
utils/export_excel.py
Helpers para exportar DataFrames a bytes .xlsx (usados en st.download_button en cada página).
"""

from io import BytesIO
import pandas as pd


def df_a_excel_bytes(dfs: dict) -> bytes:
    """
    Convierte uno o varios DataFrames a un archivo .xlsx en memoria.
    dfs: dict {nombre_hoja: DataFrame}. Permite reportes con múltiples hojas
    (ej. Módulo 4: una hoja por método).
    """
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for nombre_hoja, df in dfs.items():
            safe_name = nombre_hoja[:31]  # límite de Excel para nombres de hoja
            df.to_excel(writer, sheet_name=safe_name, index=False)
    return buffer.getvalue()

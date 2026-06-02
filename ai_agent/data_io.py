from typing import Tuple

import pandas as pd
import streamlit as st


def read_excel(uploaded_file) -> Tuple[pd.DataFrame, str]:
    excel_file = pd.ExcelFile(uploaded_file)
    sheet_names = excel_file.sheet_names

    if len(sheet_names) == 1:
        selected_sheet = sheet_names[0]
    else:
        selected_sheet = st.sidebar.selectbox("Choose sheet", sheet_names)

    dataframe = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
    return dataframe, selected_sheet


def read_data(uploaded_file) -> Tuple[pd.DataFrame, str]:
    file_name = getattr(uploaded_file, "name", "").lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(uploaded_file), "CSV"
    return read_excel(uploaded_file)

# app.py
import streamlit as st
import os, sys
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# adiciona src ao path (como no teu projeto)
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data_processor import DataProcessor
from src.models_loader import ModelLoader
from src.predictor import Predictor

# components
from components.styles import inject_styles
from components.sidebar import render_sidebar
from components.header import render_header
from components.kpis import render_kpis
from components.dataset_info import render_dataset_info
from components.prediction_chart import render_prediction_chart
from components.runner import initialize_model_loader, load_dataset, execute_prediction
from components.analysis_meta import render_analysis_meta

# Configura página
st.set_page_config(
    page_title="DeepCoin Predictor - Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Injeta CSS
inject_styles()

# Inicializa model loader
model_loader = initialize_model_loader()

# Inicializa estado do botão executar no session_state
if "execute" not in st.session_state:
    st.session_state["execute"] = False

# Renderiza sidebar e obtém configurações
sidebar_state = render_sidebar(model_loader)

# Carrega dataset
df = load_dataset(sidebar_state["file"])

if df is None or df.empty:
    st.error("❌ Não foi possível carregar os dados. Verifique se o arquivo existe na pasta 'data/'")
    st.stop()

# Aplica filtro de datas se selecionado
if sidebar_state["use_date_filter"] and sidebar_state["start_date"] and sidebar_state["end_date"]:
    df = df[(df['date'] >= pd.to_datetime(sidebar_state["start_date"])) &
            (df['date'] <= pd.to_datetime(sidebar_state["end_date"]))]
    df = df.reset_index(drop=True)

# Valida tamanho mínimo
if len(df) < sidebar_state["sequence_length"] + 10:
    st.error(f"⚠️ Dados insuficientes. Necessário pelo menos {sidebar_state['sequence_length'] + 10} registros.")
    st.stop()

# Inicializa DataFrame de predição
df_pred = None

# Executa predição somente se o botão for clicado
if st.session_state["execute"] and sidebar_state["model_choice"]:
    df_pred = execute_prediction(
        df=df,
        model_loader=model_loader,
        crypto_id=sidebar_state["crypto_id"],
        model_choice=sidebar_state["model_choice"],
        sequence_length=sidebar_state["sequence_length"],
        predict_days=sidebar_state["predict_days"]
    )
    # Reseta o estado do botão para permitir novas execuções
    st.session_state["execute"] = False

# Renderiza elementos do dashboard
render_header(df, sidebar_state)
render_kpis(df, df_pred, sidebar_state)
render_analysis_meta(df, sidebar_state)
render_prediction_chart(df, df_pred, sidebar_state)

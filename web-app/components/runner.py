# components/runner.py
import streamlit as st
import pandas as pd
import os, sys
import tensorflow as tf

# adiciona src se necessário (app.py já faz isto, mas deixamos por segurança)
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models_loader import ModelLoader
from src.data_processor import DataProcessor
from src.predictor import Predictor

@st.cache_resource
def initialize_model_loader():
    return ModelLoader(models_dir='models')

@st.cache_data
def load_dataset(filename):
    try:
        filepath = os.path.join('data', filename)
        if not os.path.exists(filepath):
            st.warning(f"⚠️ Arquivo '{filepath}' não encontrado.")
            return None
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])
        if 'close' not in df.columns:
            st.error("Coluna 'close' não encontrada no arquivo.")
            return None
        if 'volume' not in df.columns:
            df['volume'] = 0
        df['returns'] = df['close'].pct_change().fillna(0)
        df = df.sort_values('date').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dataset {filename}: {e}")
        return None

def execute_prediction(df, model_loader, crypto_id, model_choice, sequence_length, predict_days):
    df_pred = pd.DataFrame(columns=['date', 'predicted_close'])
    try:
        # Load model
        if model_choice == "Random Forest":
            model = model_loader.load_random_forest_model(crypto_id)
            model_type = "rf"
        else:
            model_type_map = {"LSTM": "lstm", "Transformer": "transformer", "GRU": "gru", "TCN": "tcn"}
            model_type = model_type_map.get(model_choice, "lstm")
            # Carrega modelo DL sem compilar (evita erro de métricas antigas)
            model = model_loader.load_deep_learning_model(crypto_id, model_type, compile=False)

            if model is not None:
                # Recompila com métricas compatíveis
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(),
                    loss='mse',
                    metrics=['mae', tf.keras.metrics.MeanSquaredError()]
                )

        if model is None:
            st.error(f"❌ Não foi possível carregar o modelo {model_choice}")
            return df_pred

        scaler = model_loader.load_scaler(crypto_id)

        data_processor = DataProcessor(sequence_length=sequence_length)
        if scaler:
            data_processor.scaler = scaler

        predictor = Predictor(model, data_processor)

        if model_choice == "Random Forest":
            df_pred = predictor.predict_random_forest(df, days_ahead=predict_days)
        else:
            df_pred = predictor.predict_future(df, days_ahead=predict_days)

        st.success("✅ Predição concluída com sucesso!")
    except Exception as e:
        st.error(f"❌ Erro ao executar predição: {e}")
        import traceback
        st.code(traceback.format_exc())
    return df_pred

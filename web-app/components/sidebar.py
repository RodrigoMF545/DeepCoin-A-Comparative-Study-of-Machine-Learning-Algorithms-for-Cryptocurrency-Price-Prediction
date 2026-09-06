# components/sidebar.py
import streamlit as st
from datetime import datetime, timedelta
from src.models_loader import ModelLoader

CRYPTO_CONFIG = {
    "BTC/USD": {"file": "btc.csv", "crypto_id": "btc", "name": "Bitcoin", "color": "#f7931a"},
    "ETH/USD": {"file": "eth.csv", "crypto_id": "eth", "name": "Ethereum", "color": "#627eea"},
    "SOL/USD": {"file": "sol.csv", "crypto_id": "sol", "name": "Solana", "color": "#14f195"}
}

def render_sidebar(model_loader: ModelLoader):
    """
    Renderiza a sidebar do dashboard com keys únicas e botão executar funcional.
    """
    with st.sidebar:
        st.image("https://i.imgur.com/8z4YH0V.png", width=120)
        st.markdown("## 🚀 DeepCoin Predictor")
        st.markdown("---")

        # Seleção da criptomoeda
        symbol = st.selectbox(
            "💰 Selecione a Criptomoeda",
            list(CRYPTO_CONFIG.keys()),
            index=0,
            key="select_crypto_sidebar"
        )
        crypto_id = CRYPTO_CONFIG[symbol]["crypto_id"]

        # Detecta modelos disponíveis
        available_models = model_loader.list_models(crypto_id)
        if available_models:
            st.success(f"✅ Modelos disponíveis para {CRYPTO_CONFIG[symbol]['name']}")

            # Extrai tipos únicos de modelo
            model_types = set()
            for f in available_models:
                if "RandomForest" in f:
                    model_types.add("Random Forest")
                elif "LSTM" in f:
                    model_types.add("LSTM")
                elif "GRU" in f:
                    model_types.add("GRU")
                elif "Transformer" in f:
                    model_types.add("Transformer")
            model_types = sorted(model_types)

            model_choice = st.selectbox(
                "🧠 Selecione o Modelo",
                model_types,
                index=0,
                key="select_model_sidebar"
            )
        else:
            st.error(f"❌ Nenhum modelo encontrado para {crypto_id}")
            model_choice = None

        st.markdown("---")
        st.markdown("### ⚙️ Configurações")
        predict_days = st.number_input(
            "Dias de Previsão", min_value=1, max_value=90, value=15, step=1, key="predict_days"
        )
        sequence_length = st.number_input(
            "Tamanho da Sequência (lookback)", min_value=30, max_value=120, value=60, step=10, key="sequence_length"
        )

        # Filtro de data
        use_date_filter = st.checkbox("📅 Filtrar por período", key="use_date_filter")
        if use_date_filter:
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "De", value=datetime.today() - timedelta(days=365), key="start_date"
                )
            with col2:
                end_date = st.date_input(
                    "Até", value=datetime.today(), key="end_date"
                )
        else:
            start_date = None
            end_date = None

        # Botão executar com session_state
        if "execute" not in st.session_state:
            st.session_state["execute"] = False

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🎯 Executar", type="primary", use_container_width=True, key="btn_execute"):
                st.session_state["execute"] = True
        with col_btn2:
            st.write("")

        execute = st.session_state["execute"]

        # Info resumida
        st.markdown("---")
        st.markdown("### 📌 Informações")
        st.markdown(f"""
        **Modelos Disponíveis:** LSTM, Transformer, GRU, Random Forest  
        **Arquivos necessários:** models/{crypto_id}/best_model_*.h5 ou .joblib
        """)

    return {
        "symbol": symbol,
        "crypto_id": crypto_id,
        "file": CRYPTO_CONFIG[symbol]["file"],
        "name": CRYPTO_CONFIG[symbol]["name"],
        "color": CRYPTO_CONFIG[symbol]["color"],
        "model_choice": model_choice,
        "predict_days": predict_days,
        "sequence_length": sequence_length,
        "use_date_filter": use_date_filter,
        "start_date": start_date,
        "end_date": end_date,
        "execute": execute
    }

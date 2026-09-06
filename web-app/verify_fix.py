import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf

# Add current directory to path so we can import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models_loader import ModelLoader
from src.data_processor import DataProcessor
from src.predictor import Predictor

def test_inference_pipeline():
    print("🚀 Iniciando verificação do pipeline de inferência...")
    
    # 1. Setup
    crypto_id = 'btc'
    model_loader = ModelLoader(models_dir='models')
    
    # 2. Carrega Scaler
    print(f"📂 Carregando scaler para {crypto_id}...")
    scaler = model_loader.load_scaler(crypto_id)
    if scaler is None:
        print("❌ Scaler não encontrado!")
        return
    print(f"✅ Scaler carregado: {type(scaler)}")
    
    # 3. Carrega Modelo
    print(f"📂 Carregando modelo LSTM para {crypto_id}...")
    model = model_loader.load_deep_learning_model(crypto_id, 'lstm', compile=False)
    if model is None:
        print("❌ Modelo não encontrado!")
        return
    print(f"✅ Modelo carregado. Input shape: {model.input_shape}")
    
    # 4. Carrega Dados
    print("📂 Carregando dados de teste...")
    try:
        df = pd.read_csv(os.path.join('data', f'{crypto_id}.csv'))
        df['date'] = pd.to_datetime(df['date'])
        print(f"✅ Dados carregados. Shape: {df.shape}")
    except Exception as e:
        print(f"❌ Erro ao carregar dados: {e}")
        return

    # 5. Configura Processador
    dp = DataProcessor(sequence_length=60)
    dp.scaler = scaler
    
    # 6. Inicializa Preditor
    predictor = Predictor(model, dp)
    
    # 7. Executa Predição
    print("🔮 Executando predict_future(days_ahead=30)...")
    try:
        pred_df = predictor.predict_future(df, days_ahead=30, use_iterative=True)
        
        if pred_df.empty:
            print("❌ Predição retornou DataFrame vazio!")
            return
            
        print("\n📊 Resultados da Predição:")
        print(pred_df.head())
        
        # 8. Validação de Dinamismo
        preds = pred_df['predicted_close'].values
        std_dev = np.std(preds)
        price_change = preds[-1] - preds[0]
        
        print(f"\n📈 Estatísticas:")
        print(f"   Média: {np.mean(preds):.2f}")
        print(f"   Desvio Padrão: {std_dev:.4f}")
        print(f"   Delta (Fim - Início): {price_change:.2f}")
        
        if std_dev < 0.01: # Check for extremely flat line
            print("\n⚠️ ALERTA: Predições parecem estáticas (linha reta)!")
            print("   Motivo possível: Heurística de features não está funcionando ou modelo colapsou.")
        else:
            print("\n✅ SUCESSO: Predições dinâmicas detectadas.")
            
    except Exception as e:
        print(f"❌ Erro durante predição: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference_pipeline()

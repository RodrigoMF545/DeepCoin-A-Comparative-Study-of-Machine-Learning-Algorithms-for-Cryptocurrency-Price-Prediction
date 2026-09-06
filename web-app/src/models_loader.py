# src/models_loader.py
import os
import joblib
import tensorflow as tf

class ModelLoader:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir

    def list_models(self, crypto_id):
        """Lista modelos disponíveis para uma criptomoeda"""
        path = os.path.join(self.models_dir, crypto_id)
        if not os.path.exists(path):
            return []
        return [f for f in os.listdir(path) if f.endswith(('.h5', '.joblib'))]

    def load_random_forest_model(self, crypto_id):
        """Carrega modelo Random Forest (.joblib)"""
        path = os.path.join(self.models_dir, crypto_id)
        files = [f for f in os.listdir(path) if 'RandomForest' in f and f.endswith('.joblib')]
        if not files:
            return None
        return joblib.load(os.path.join(path, files[0]))

    def load_deep_learning_model(self, crypto_id, model_type='lstm', compile=True):
        """Carrega modelo DL (.h5) com opção de não compilar (para evitar erro de métricas antigas)"""
        path = os.path.join(self.models_dir, crypto_id)
        model_map = {
            'lstm': 'LSTM',
            'gru': 'GRU',
            'tcn': 'TCN',
            'transformer': 'Transformer'
        }
        type_str = model_map.get(model_type.lower(), 'LSTM')
        files = [f for f in os.listdir(path) if type_str in f and f.endswith('.h5')]
        if not files:
            return None
        return tf.keras.models.load_model(os.path.join(path, files[0]), compile=compile)

    def load_scaler(self, crypto_id):
        """Carrega scaler (.joblib)"""
        path = os.path.join(self.models_dir, crypto_id)
        files = [f for f in os.listdir(path) if 'scaler' in f.lower() and f.endswith('.joblib')]
        if not files:
            return None
        return joblib.load(os.path.join(path, files[0]))

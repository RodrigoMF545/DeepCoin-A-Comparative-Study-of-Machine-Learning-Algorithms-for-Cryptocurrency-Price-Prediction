"""
Data Processor Module
Processa e prepara dados para predição
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

class DataProcessor:
    def __init__(self, sequence_length=60):
        """
        Args:
            sequence_length: Número de timesteps para sequências
        """
        self.sequence_length = sequence_length
        self.scaler = None
        
    def load_scaler(self, scaler_path):
        """Carrega scaler salvo"""
        try:
            self.scaler = joblib.load(scaler_path)
            return True
        except Exception as e:
            print(f"Erro ao carregar scaler: {e}")
            return False
    
    def prepare_data(self, df, features=['close', 'volume', 'open', 'high', 'low']):
        """
        Prepara dados para predição
        
        Args:
            df: DataFrame com dados históricos
            features: Lista de features a usar
            
        Returns:
            X: Dados preparados para modelo
            dates: Datas correspondentes
        """
        # Seleciona features disponíveis
        available_features = [f for f in features if f in df.columns]
        
        if not available_features:
            print("⚠️ Nenhuma feature encontrada no DataFrame!")
            return np.array([]), np.array([])
        
        # Valida features contra o scaler se estiver carregado
        if self.scaler is not None and hasattr(self.scaler, 'n_features_in_'):
            if len(available_features) != self.scaler.n_features_in_:
                print(f"⚠️ Aviso: Scaler espera {self.scaler.n_features_in_} features, mas {len(available_features)} foram fornecidas.")
                # Tenta ajustar se possível ou falha? Por enquanto apenas avisamos.
        
        # Extrai dados
        data = df[available_features].values
        dates = df['date'].values
        
        # Normaliza dados
        if self.scaler is None:
            print("⚠️ ERRO CRÍTICO: Scaler não carregado! Não é possível normalizar corretamente para inferência.")
            # Em produção, isso deve ser um erro fatal. Para debug, retornamos vazio.
            return np.array([]), np.array([])
        else:
            try:
                data_scaled = self.scaler.transform(data)
            except Exception as e:
                print(f"❌ Erro na normalização: {e}")
                return np.array([]), np.array([])
        
        # Cria sequências
        X = []
        valid_dates = []
        
        if len(data_scaled) <= self.sequence_length:
             print(f"⚠️ Dados insuficientes: {len(data_scaled)} <= {self.sequence_length}")
             return np.array([]), np.array([])

        for i in range(len(data_scaled) - self.sequence_length):
            X.append(data_scaled[i:i + self.sequence_length])
            valid_dates.append(dates[i + self.sequence_length])
        
        return np.array(X), np.array(valid_dates)
    
    def prepare_last_sequence(self, df, features=['close', 'volume', 'open', 'high', 'low']):
        """
        Prepara última sequência para predição futura
        
        Args:
            df: DataFrame com dados históricos
            features: Lista de features a usar
            
        Returns:
            Última sequência preparada
        """
        available_features = [f for f in features if f in df.columns]
        
        if not available_features:
            raise ValueError("Features não encontradas no DataFrame")
        
        # Pega últimos sequence_length pontos
        if len(df) < self.sequence_length:
             raise ValueError(f"Dados insuficientes ({len(df)}) para sequence_length={self.sequence_length}")

        data = df[available_features].tail(self.sequence_length).values
        
        # Normaliza
        if self.scaler is not None:
            data_scaled = self.scaler.transform(data)
        else:
            raise ValueError("Scaler não inicializado! Carregue o scaler antes de preparar a sequência.")
        
        return data_scaled.reshape(1, self.sequence_length, len(available_features))
    
    def inverse_transform_predictions(self, predictions, feature_index=0):
        """
        Reverte normalização das predições
        
        Args:
            predictions: Predições normalizadas
            feature_index: Índice da feature (0 para close)
            
        Returns:
            Predições em escala original
        """
        if self.scaler is None:
            return predictions
        
        # Cria array com shape correto
        dummy = np.zeros((len(predictions), self.scaler.n_features_in_))
        dummy[:, feature_index] = predictions
        
        # Inverte transformação
        inversed = self.scaler.inverse_transform(dummy)
        
        return inversed[:, feature_index]
    
    def calculate_technical_indicators(self, df):
        """
        Calcula indicadores técnicos
        
        Args:
            df: DataFrame com dados
            
        Returns:
            DataFrame com indicadores adicionados
        """
        df = df.copy()
        
        # Moving Averages
        df['MA7'] = df['close'].rolling(window=7).mean()
        df['MA21'] = df['close'].rolling(window=21).mean()
        df['MA50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['BB_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        
        # Volatility
        df['volatility'] = df['close'].pct_change().rolling(window=14).std()
        
        return df
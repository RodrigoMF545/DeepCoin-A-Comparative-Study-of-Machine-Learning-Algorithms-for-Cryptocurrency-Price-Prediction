

"""
Predictor Module - Implementação Completa
Gera predições usando modelos treinados de Deep Learning e ML
"""

import pandas as pd
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

class Predictor:
    def __init__(self, model, data_processor):
        """
        Inicializa o preditor
        
        Args:
            model: Modelo treinado (Keras ou Sklearn)
            data_processor: Instância de DataProcessor para normalização
        """
        self.model = model
        self.data_processor = data_processor
        self.is_sklearn_model = hasattr(model, 'predict') and not hasattr(model, 'predict_classes')
    
    def predict_future(self, df, days_ahead=15, use_iterative=True):
        """
        Gera predições futuras usando modelos de Deep Learning
        
        Args:
            df: DataFrame com dados históricos
            days_ahead: Número de dias para prever
            use_iterative: (Descontinuado) Sempre usa iterativo agora.
            
        Returns:
            DataFrame com predições [date, predicted_close]
        """
        try:
            return self._predict_future_iterative(df, days_ahead)
        except Exception as e:
            print(f"❌ Erro ao gerar predições futuras: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(columns=['date', 'predicted_close'])
    
    def _predict_future_iterative(self, df, days_ahead):
        """
        Predição iterativa com atualização heurística de features.
        
        Como o modelo é multivariado (usa Open, High, Low, Volume) mas prediz apenas Close,
        precisamos estimar as outras features para os passos futuros.
        
        Estratégia:
        - Open(t+1) = Close(t)
        - High(t+1) ~ Close(t) + Volatilidade
        - Low(t+1) ~ Close(t) - Volatilidade
        - Volume(t+1) ~ Média móvel recente
        """
        # Prepara última sequência
        # Shape: (1, sequence_length, n_features)
        last_sequence = self.data_processor.prepare_last_sequence(df)
        
        predictions = []
        dates = []
        
        current_sequence = last_sequence.copy()
        last_date = pd.to_datetime(df['date'].iloc[-1])
        
        # Calcula estatísticas recentes para estimar features futuras
        # Precisamos dos dados originais para calcular volatilidade real, não a escalada
        recent_data = df.tail(self.data_processor.sequence_length)
        if 'close' in recent_data.columns:
            # Volatilidade média dos últimos dias (absoluta)
            recent_volatility = (recent_data['high'] - recent_data['low']).mean()
            # Volume médio
            recent_volume_mean = recent_data['volume'].mean() if 'volume' in recent_data.columns else 0
        else:
            recent_volatility = 0
            recent_volume_mean = 0
            
        # Para normalização das features estimadas, precisamos do scaler se ele existir
        scaler = self.data_processor.scaler
        
        print(f"🔮 Gerando {days_ahead} predições iterativas (com heurística)...")
        
        for i in range(days_ahead):
            # 1. Predição para o próximo timestep (t+1) baseada na sequência atual
            # Saída do model.predict depende do modelo, mas geralmente é (1, 1) ou (1, n_outputs)
            pred_scaled = self.model.predict(current_sequence, verbose=0)
            
            # Extrai valor escalar predito (scaled close)
            # Flatten para pegar o primeiro elemento independente do shape
            pred_value_scaled = pred_scaled.flatten()[0]
            
            predictions.append(pred_value_scaled)
            
            # 2. Calcula data
            next_date = last_date + timedelta(days=i+1)
            dates.append(next_date)
            
            # 3. Atualiza sequência para próxima predição
            # Precisamos criar um novo timestep com features coerentes
            
            # Recupera o último timestep da sequência atual para referência (valores escalados)
            last_timestep_scaled = current_sequence[0, -1, :]
            
            # Desnormaliza o último close (assumindo que close é índice 0)
            # Mas espera! Temos `pred_value_scaled` que é o NOVO close predito.
            # Vamos estimar as próximas features baseadas nesse novo close.
            
            # Primeiro, precisamos desnormalizar o pred_value_scaled para calcular High/Low coerentes
            # Nota: Isso é custoso fazer em loop, mas necessário para precisão.
            if scaler:
                # Cria dummy array para inverse_transform
                dummy = np.zeros((1, scaler.n_features_in_))
                dummy[0, 0] = pred_value_scaled # Close
                # Preenche resto com 0 só pra rodar, mas o ideal seria usar médias
                inversed = scaler.inverse_transform(dummy)[0]
                pred_close_original = inversed[0]
            else:
                pred_close_original = pred_value_scaled

            # Estima features originais para t+1
            next_open = pred_close_original # Open abre no valor do último Close
            next_high = pred_close_original + recent_volatility # Estima High
            next_low = pred_close_original - recent_volatility  # Estima Low
            next_volume = recent_volume_mean
            
            # Monta vetor de features originais (Ordem deve bater com treinamento!)
            # Assumindo ordem padrão: Close, Volume, Open, High, Low
            # Se a ordem for diferente, isso vai quebrar. Vamos tentar ser agnósticos ou assumir o padrão do DataProcessor.
            # DataProcessor padrão: predictions, feature_index=0 -> Close.
            # Vamos assumir que a ordem das features no scaler e no DataProcessor são consistentes.
            
            # CRÍTICO: Precisamos saber a ordem das features.
            # O DataProcessor usa: available_features = [f for f in features if f in df.columns]
            # Features default: ['close', 'volume', 'open', 'high', 'low']
            
            # Vamos construir um vetor novo.
            # Se não soubermos a ordem exata, a melhor aposta é manter a lógica estatística simples:
            # - Feature 0 (Close): Novo valor predito
            # - Outras features: Copiar do último passo OU aplicar heurística se soubermos qual é qual
            
            # Abordagem Híbrida Segura:
            # 1. Feature 0 é SEMPRE Close (assumido).
            # 2. Tenta inferir outras features pelo DataProcessor.scaler se possível, ou usa cópia com decaimento.
            
            new_timestep_features = np.zeros_like(last_timestep_scaled)
            new_timestep_features[0] = pred_value_scaled # Feature 0 = Close
            
            # Se tivermos mais features, tentamos preencher
            if len(new_timestep_features) > 1:
                # Copia as outras features do passo anterior como base
                new_timestep_features[1:] = last_timestep_scaled[1:]
                
                # Se tivermos acesso ao scaler para re-normalizar, podemos fazer melhor
                if scaler:
                    # Tenta construir o vetor feature original
                    next_features_dict = {
                        'close': next_open, # O próximo 'close' input é na verdade o preço atual que acabou de fechar? 
                                            # Não, em série temporal: input(t) -> predict(t+1).
                                            # Para prever t+2, input(t+1) deve conter o Close(t+1) que acabamos de prever.
                        'open': next_open,
                        'high': next_high,
                        'low': next_low,
                        'volume': next_volume
                    }
                    
                    # Precisamos da ordem das features do scaler
                    if hasattr(scaler, 'feature_names_in_'):
                        feature_names = scaler.feature_names_in_
                        # Constrói vetor ordenado
                        vector_original = []
                        for name in feature_names:
                            key = name.lower()
                            if key in next_features_dict:
                                vector_original.append(next_features_dict[key])
                            else:
                                vector_original.append(next_open) # Fallback seguro
                        
                        # Normaliza esse vetor
                        vector_original = np.array(vector_original).reshape(1, -1)
                        new_timestep_features = scaler.transform(vector_original)[0]
            
            # Remonta shape (1, 1, n_features)
            new_timestep = new_timestep_features.reshape(1, 1, len(new_timestep_features))
            
            # Desloca a sequência: remove primeiro timestep, adiciona novo
            current_sequence = np.concatenate([
                current_sequence[:, 1:, :],
                new_timestep
            ], axis=1)
        
        # Desnormaliza predições finais
        predictions_original = self.data_processor.inverse_transform_predictions(
            np.array(predictions), feature_index=0
        )
        
        # Cria DataFrame
        df_pred = pd.DataFrame({
            'date': dates,
            'predicted_close': predictions_original
        })
        
        print(f"✅ {len(df_pred)} predições geradas!")
        return df_pred
    
    def predict_random_forest(self, df, days_ahead=15):
        """
        Gera predições futuras usando Random Forest
        
        Args:
            df: DataFrame com dados históricos
            days_ahead: Número de dias para prever
            
        Returns:
            DataFrame com predições
        """
        try:
            sequence_length = self.data_processor.sequence_length
            
            # Pega últimos dados (sem normalização para RF)
            last_data = df['close'].tail(sequence_length).values
            
            predictions = []
            dates = []
            
            current_data = list(last_data)
            last_date = pd.to_datetime(df['date'].iloc[-1])
            
            print(f"🌲 Gerando {days_ahead} predições com Random Forest...")
            
            for i in range(days_ahead):
                # Prepara input (últimos sequence_length valores)
                X = np.array(current_data[-sequence_length:]).reshape(1, -1)
                
                # Predição
                pred = self.model.predict(X)[0]
                predictions.append(pred)
                
                # Próxima data
                next_date = last_date + timedelta(days=i+1)
                dates.append(next_date)
                
                # Atualiza dados (adiciona predição ao histórico)
                current_data.append(pred)
            
            # Cria DataFrame
            df_pred = pd.DataFrame({
                'date': dates,
                'predicted_close': predictions
            })
            
            print(f"✅ {len(df_pred)} predições RF geradas!")
            return df_pred
            
        except Exception as e:
            print(f"❌ Erro ao gerar predições RF: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame(columns=['date', 'predicted_close'])
    
    def predict_historical(self, df, verbose=True):
        """
        Gera predições para dados históricos (para validação do modelo)
        
        Args:
            df: DataFrame com dados históricos
            verbose: Se True, mostra progresso
            
        Returns:
            Tuple (y_true, y_pred, dates) ou (None, None, None) em caso de erro
        """
        try:
            # Prepara dados
            X, dates = self.data_processor.prepare_data(df)
            
            if len(X) == 0:
                print("⚠️ Nenhuma sequência válida para predição")
                return None, None, None
            
            if verbose:
                print(f"📊 Gerando predições históricas para {len(X)} amostras...")
            
            # Gera predições em lotes para eficiência
            batch_size = 32
            predictions = []
            
            for i in range(0, len(X), batch_size):
                batch = X[i:i+batch_size]
                batch_pred = self.model.predict(batch, verbose=0)
                predictions.extend(batch_pred)
            
            predictions = np.array(predictions)
            
            # Desnormaliza predições
            predictions_flat = predictions.flatten()
            predictions_original = self.data_processor.inverse_transform_predictions(
                predictions_flat, feature_index=0
            )
            
            # Valores reais correspondentes
            actual_values = df['close'].iloc[self.data_processor.sequence_length:].values
            
            # Garante mesmo tamanho (importante!)
            min_len = min(len(actual_values), len(predictions_original), len(dates))
            
            y_true = actual_values[:min_len]
            y_pred = predictions_original[:min_len]
            dates_valid = dates[:min_len]
            
            if verbose:
                print(f"✅ Predições históricas geradas: {min_len} amostras")
            
            return y_true, y_pred, dates_valid
            
        except Exception as e:
            print(f"❌ Erro ao gerar predições históricas: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None
    
    def predict_with_confidence(self, df, days_ahead=15, n_simulations=10):
        """
        Gera predições com intervalos de confiança usando Monte Carlo
        
        Args:
            df: DataFrame com dados históricos
            days_ahead: Número de dias para prever
            n_simulations: Número de simulações para estimar incerteza
            
        Returns:
            DataFrame com predições e intervalos de confiança
        """
        try:
            print(f"🎲 Gerando predições com {n_simulations} simulações...")
            
            all_predictions = []
            
            for i in range(n_simulations):
                # Adiciona pequeno ruído aos dados para simular incerteza
                df_noisy = df.copy()
                noise = np.random.normal(0, df['close'].std() * 0.01, len(df))
                df_noisy['close'] = df_noisy['close'] + noise
                
                # Gera predição
                pred_df = self.predict_future(df_noisy, days_ahead, use_iterative=True)
                all_predictions.append(pred_df['predicted_close'].values)
            
            # Calcula estatísticas
            all_predictions = np.array(all_predictions)
            
            dates = self.predict_future(df, days_ahead)['date']
            mean_pred = np.mean(all_predictions, axis=0)
            std_pred = np.std(all_predictions, axis=0)
            
            # Intervalo de confiança 95%
            lower_bound = mean_pred - 1.96 * std_pred
            upper_bound = mean_pred + 1.96 * std_pred
            
            df_pred = pd.DataFrame({
                'date': dates,
                'predicted_close': mean_pred,
                'lower_95': lower_bound,
                'upper_95': upper_bound,
                'std': std_pred
            })
            
            print(f"✅ Predições com confiança geradas!")
            return df_pred
            
        except Exception as e:
            print(f"❌ Erro ao gerar predições com confiança: {e}")
            return pd.DataFrame(columns=['date', 'predicted_close', 'lower_95', 'upper_95'])
    
    def evaluate_on_test_set(self, df, test_size=0.2):
        """
        Avalia o modelo em um conjunto de teste
        
        Args:
            df: DataFrame com dados completos
            test_size: Proporção dos dados para teste (0.2 = 20%)
            
        Returns:
            Dict com métricas de avaliação
        """
        try:
            # Divide dados
            split_idx = int(len(df) * (1 - test_size))
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()
            
            print(f"📊 Avaliando modelo...")
            print(f"   Treino: {len(train_df)} amostras")
            print(f"   Teste: {len(test_df)} amostras")
            
            # Gera predições para período de teste
            y_true, y_pred, dates = self.predict_historical(test_df, verbose=False)
            
            if y_true is None:
                return None
            
            # Calcula métricas
            from src.metricsCalculator import MetricsCalculator
            metrics = MetricsCalculator.calculate_all_metrics(y_true, y_pred)
            
            print(f"✅ Avaliação completa!")
            return metrics
            
        except Exception as e:
            print(f"❌ Erro na avaliação: {e}")
            return None

import os
import time
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks, models, Input
from tensorflow.keras.layers import Conv1D, Dense, Lambda, Add, MultiHeadAttention, LayerNormalization, Dropout, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

# Import Custom Layers
from src.custom_layers import TransformerBlock, PositionalEncoding

warnings.filterwarnings("ignore")

# Define random seeds
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

class ModelFactory:
    @staticmethod
    def compile_model(model, lr=1e-3):
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss='mse',
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
        )
        return model

    @staticmethod
    def build_random_forest(input_shape):
        # input_shape not strictly used for init, but keeping signature consistent
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        return model

    @staticmethod
    def build_lstm(input_shape):
        model = keras.Sequential([
            layers.LSTM(100, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.LSTM(100, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dense(1)
        ])
        return ModelFactory.compile_model(model)

    @staticmethod
    def build_gru(input_shape):
        model = keras.Sequential([
            layers.GRU(100, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.3),
            layers.GRU(100, return_sequences=False),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
            layers.Dense(1)
        ])
        return ModelFactory.compile_model(model)

    @staticmethod
    def build_tcn(input_shape):
        inputs = Input(shape=input_shape)
        x = Conv1D(filters=64, kernel_size=1, padding='causal', activation='relu')(inputs)
        
        def tcn_block(x, dilation_rate, filters=64, kernel_size=3):
            conv = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(x)
            conv = layers.Dropout(0.2)(conv)
            conv = layers.Conv1D(filters, kernel_size, padding='causal', dilation_rate=dilation_rate, activation='relu')(conv)
            if x.shape[-1] != filters:
                x = Conv1D(filters, 1, padding='same')(x)
            return Add()([x, conv])
        
        for dilation in [1, 2, 4, 8, 16]:
            x = tcn_block(x, dilation)
        
        x = layers.Dense(32, activation='relu')(x[:, -1, :])
        outputs = layers.Dense(1)(x)
        model = Model(inputs=inputs, outputs=outputs)
        return ModelFactory.compile_model(model)

    @staticmethod
    def build_transformer(input_shape, heads=4, ff_dim=128):
        inputs = layers.Input(shape=input_shape)
        x = layers.Dense(64)(inputs)
        x = PositionalEncoding(sequence_length=input_shape[0], d_model=64)(x)
        x = TransformerBlock(embed_dim=64, num_heads=heads, ff_dim=ff_dim)(x, training=True)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)
        return ModelFactory.compile_model(model)

class Trainer:
    def __init__(self, data_dir='data', models_dir='models'):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.feature_cols = ['open', 'high', 'low', 'close', 'volume']
        self.window_size = 60
        self.model_builders = {
            'RandomForest': ModelFactory.build_random_forest,
            'LSTM': ModelFactory.build_lstm,
            'GRU': ModelFactory.build_gru,
            'TCN': ModelFactory.build_tcn,
            'Transformer': ModelFactory.build_transformer
        }

    def create_sequences(self, data, target_col_idx):
        X, y = [], []
        for i in range(self.window_size, len(data)):
            X.append(data[i - self.window_size:i, :])
            y.append(data[i, target_col_idx])
        return np.array(X), np.array(y)

    def flatten_3d_to_2d(self, X):
        return X.reshape(X.shape[0], -1)

    def run(self, epochs=50, batch_size=32):
        # List all csv files
        if not os.path.exists(self.data_dir):
            print(f"❌ Syntax Error: Data directory '{self.data_dir}' not found.")
            return

        files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        for filename in files:
            crypto_id = filename.replace('.csv', '') # e.g. 'btc'
            filepath = os.path.join(self.data_dir, filename)
            
            print(f"\nProcessing {crypto_id} from {filepath}...")
            
            # Load and preprocess
            try:
                df = pd.read_csv(filepath)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values("date").reset_index(drop=True)
                
                # Check features
                missing = [c for c in self.feature_cols if c not in df.columns]
                if missing:
                    print(f"⚠️ Missing columns {missing} in {filename}. Skipping.")
                    continue
                
                # Prepare data
                data_array = df[self.feature_cols].values
                
                # Train/Test split
                train_size = int(len(data_array) * 0.8)
                train_data = data_array[:train_size]
                test_data = data_array[train_size:]
                
                # Scale
                scaler = MinMaxScaler()
                scaled_train = scaler.fit_transform(train_data)
                scaled_test = scaler.transform(test_data)
                
                # Save Scaler
                save_path = os.path.join(self.models_dir, crypto_id)
                os.makedirs(save_path, exist_ok=True)
                scaler_name = f"scaler_{crypto_id}_v2.joblib"
                joblib.dump(scaler, os.path.join(save_path, scaler_name))
                print(f"✅ Saved scaler to {scaler_name}")
                
                # Create sequences
                target_col_idx = self.feature_cols.index('close')
                X_train, y_train = self.create_sequences(scaled_train, target_col_idx)
                X_test, y_test = self.create_sequences(scaled_test, target_col_idx)
                
                print(f"   Train shape: {X_train.shape}, Test shape: {X_test.shape}")
                
                # Train all models
                for model_name, builder in self.model_builders.items():
                    print(f"\n   🚀 Training {model_name}...")
                    
                    model = builder(X_train.shape[1:])
                    
                    if model_name == 'RandomForest':
                        X_train_rf = self.flatten_3d_to_2d(X_train)
                        X_test_rf = self.flatten_3d_to_2d(X_test)
                        
                        model.fit(X_train_rf, y_train)
                        
                        # Save RF
                        model_filename = f"best_model_{model_name}_{crypto_id}.joblib"
                        joblib.dump(model, os.path.join(save_path, model_filename))
                        
                        # Evaluate
                        y_pred = model.predict(X_test_rf)
                        
                    else: # Deep Learning
                        # Log time
                        start_time = time.time()
                        
                        history = model.fit(
                            X_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_data=(X_test, y_test),
                            verbose=1,
                            callbacks=[
                                callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                                callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
                            ]
                        )
                        
                        # Save DL model
                        model_filename = f"best_model_{model_name}_{crypto_id}.h5"
                        model.save(os.path.join(save_path, model_filename))
                        
                        # Evaluate
                        y_pred = model.predict(X_test).flatten()
                        
                    # Calculate Metrics (Inverse Transform first)
                    # We need to reconstruct the full feature set to inverse transform
                    # Because scaler expects 5 features
                    
                    def inverse_y(y_scaled):
                        dummy = np.zeros((len(y_scaled), len(self.feature_cols)))
                        dummy[:, target_col_idx] = y_scaled
                        return scaler.inverse_transform(dummy)[:, target_col_idx]
                    
                    y_test_real = inverse_y(y_test)
                    y_pred_real = inverse_y(y_pred)
                    
                    mae = mean_absolute_error(y_test_real, y_pred_real)
                    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
                    r2 = r2_score(y_test_real, y_pred_real)
                    
                    print(f"   🏆 {model_name} Results: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")
                    print(f"   💾 Model saved to {model_filename}")

            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
                import traceback
                traceback.print_exc()

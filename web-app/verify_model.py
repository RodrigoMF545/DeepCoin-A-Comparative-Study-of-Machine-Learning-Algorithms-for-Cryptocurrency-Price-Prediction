import os
import sys
import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
from src.data_processor import DataProcessor

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
# We need to find a crypto ID that exists. I saw 'btc' earlier.
CRYPTO_ID = 'btc'

def check_model_and_scaler():
    print(f"Checking models for {CRYPTO_ID} in {MODELS_DIR}...")
    
    # 1. Load Scaler
    scaler_path = os.path.join(MODELS_DIR, CRYPTO_ID)
    if not os.path.exists(scaler_path):
        print(f"❌ Path not found: {scaler_path}")
        return

    scaler_files = [f for f in os.listdir(scaler_path) if 'scaler' in f.lower() and f.endswith('.joblib')]
    
    if not scaler_files:
        print("❌ No scaler found!")
        return
    
    scaler_file = os.path.join(scaler_path, scaler_files[0])
    print(f"✅ Found scaler: {scaler_files[0]}")
    
    try:
        scaler = joblib.load(scaler_file)
        print(f"   Type: {type(scaler)}")
        print(f"   n_features_in_: {getattr(scaler, 'n_features_in_', 'N/A')}")
        print(f"   feature_names_in_: {getattr(scaler, 'feature_names_in_', 'N/A')}")
        if hasattr(scaler, 'data_min_'):
            print(f"   min_: {scaler.data_min_}")
        if hasattr(scaler, 'data_max_'):
            print(f"   max_: {scaler.data_max_}")
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        return

    # 2. Load Model (LSTM)
    model_files = [f for f in os.listdir(scaler_path) if 'LSTM' in f and f.endswith('.h5')]
    if not model_files:
        print("❌ No LSTM model found!")
        return

    model_file = os.path.join(scaler_path, model_files[0])
    print(f"\n✅ Found model: {model_files[0]}")
    
    try:
        model = tf.keras.models.load_model(model_file, compile=False)
        # model.input_shape might be (None, 60, 5)
        print(f"   Model Input Shape: {model.input_shape}")
        
        # The last dimension is features
        expected_features = model.input_shape[-1]
        
        if hasattr(scaler, 'n_features_in_'):
            if expected_features != scaler.n_features_in_:
                print(f"❌ MISMATCH: Model expects {expected_features} features, scaler has {scaler.n_features_in_}")
            else:
                print(f"✅ Feature count matches: {expected_features}")
        else:
            print(f"⚠️ Scaler does not report n_features_in_ (might be old sklearn version or not fitted)")
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Test DataProcessor
    print("\nTesting DataProcessor...")
    try:
        dp = DataProcessor(sequence_length=60)
        # Manually inject scaler
        dp.scaler = scaler
        
        # Create dummy data with 5 columns + date
        # Assuming typical order: close, open, high, low, volume
        # Or maybe close, volume, open, high, low?
        # The feature_names_in_ from scaler will tell us the expected order!
        
        dates = pd.date_range(start='2023-01-01', periods=100)
        
        # Generate random data
        data_dict = {
            'date': dates,
            'close': np.random.rand(100) * 1000 + 20000,
            'volume': np.random.rand(100) * 1000,
            'open': np.random.rand(100) * 1000 + 20000,
            'high': np.random.rand(100) * 1000 + 20000,
            'low': np.random.rand(100) * 1000 + 20000
        }
        df = pd.DataFrame(data_dict)
        
        # We need to know which features DataProcessor selects by default
        # prepare_last_sequence default features = ['close', 'volume', 'open', 'high', 'low']
        
        print(f"   Calling prepare_last_sequence with default features...")
        seq = dp.prepare_last_sequence(df)
        print(f"   Produced sequence shape: {seq.shape}")
        
        pred = model.predict(seq, verbose=0)
        print(f"   Prediction raw (scaled): {pred}")
        
        pred_val = pred[0,0] if len(pred.shape) > 1 else pred[0]
        
        # Test inverse transform
        print(f"   Testing inverse transform...")
        inv = dp.inverse_transform_predictions(np.array([pred_val]))
        print(f"   Inversed prediction: {inv}")
        
    except Exception as e:
        print(f"❌ Error in DataProcessor flow: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_model_and_scaler()

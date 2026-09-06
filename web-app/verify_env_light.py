import os
import joblib
import pandas as pd
import numpy as np

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
CRYPTO_ID = 'btc'

def check_scaler_only():
    print(f"Checking scaler for {CRYPTO_ID} in {MODELS_DIR}...")
    
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
            
        # Check scale
        if hasattr(scaler, 'scale_'):
            print(f"   scale_: {scaler.scale_}")
            
    except Exception as e:
        print(f"❌ Error loading scaler: {e}")
        return

if __name__ == "__main__":
    check_scaler_only()

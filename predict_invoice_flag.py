import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/predict_flag_invoice.pkl"

def predict_invoice_flag(input_data):
    """
    Predict invoice flag for manual approval.
    
    Parameters
    ----------
    input_data : dict
        Dictionary with invoice features
    
    Returns
    -------
    pd.DataFrame with Predicted_Flag column
    """
    try:
        model = joblib.load(MODEL_PATH)
        input_df = pd.DataFrame(input_data)
        
        # Make prediction
        predictions = model.predict(input_df)
        
        # Create result DataFrame with prediction
        result_df = input_df.copy()
        result_df['Predicted_Flag'] = predictions
        
        return result_df
    except FileNotFoundError:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

if __name__ == "__main__":
    pass
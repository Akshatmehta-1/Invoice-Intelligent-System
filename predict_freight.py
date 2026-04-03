import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/predict_freight_model.pkl"

def predict_freight_cost(input_data):
    """
    Predict freight cost for vendor invoices.
    
    Parameters
    ----------
    input_data : dict
        Dictionary with 'Dollars' key
    
    Returns
    -------
    pd.DataFrame with Predicted_Freight column
    """
    try:
        model = joblib.load(MODEL_PATH)
        input_df = pd.DataFrame(input_data)
        
        # Make prediction
        predictions = model.predict(input_df)
        
        # Create result DataFrame with prediction
        result_df = input_df.copy()
        result_df['Predicted_Freight'] = predictions
        
        return result_df
    except FileNotFoundError:
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

if __name__ == "__main__":
    pass
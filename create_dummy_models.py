"""
Script to create dummy trained models for testing purposes.
These are simple models trained on synthetic data.
Replace with real trained models when actual data is available.
"""

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import os

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

print("Creating dummy models...")

# ========================================================================
# Model 1: Freight Cost Prediction (Regression)
# ========================================================================
print("\n1️⃣  Creating Freight Cost Predictor...")

# Synthetic training data
X_freight = np.array([
    [1000],
    [1500],
    [2000],
    [2500],
    [3000],
    [3500],
    [4000],
    [4500],
    [5000],
    [5500],
]).astype(float)

y_freight = np.array([50, 75, 100, 125, 150, 175, 200, 225, 250, 275])

# Train Random Forest Regressor
freight_model = RandomForestRegressor(
    n_estimators=10,
    max_depth=5,
    random_state=42
)
freight_model.fit(X_freight, y_freight)

# Save model
joblib.dump(freight_model, 'models/predict_freight_model.pkl')
print("✅ Freight model saved: models/predict_freight_model.pkl")

# ========================================================================
# Model 2: Invoice Flag Prediction (Classification)
# ========================================================================
print("\n2️⃣  Creating Invoice Flag Classifier...")

# Synthetic training data
# Features: invoice_quantity, invoice_dollars, freight, total_item_quantity, total_item_dollars
X_flag = np.array([
    [50, 352.95, 1.73, 162, 2476.0],      # Flag=0 (Safe)
    [60, 400.00, 2.00, 170, 2600.0],      # Flag=1 (Risky - mismatch)
    [70, 450.00, 2.50, 180, 2800.0],      # Flag=0 (Safe)
    [45, 320.00, 1.50, 140, 2300.0],      # Flag=0 (Safe)
    [80, 2000.00, 10.0, 100, 200.0],      # Flag=1 (Risky - large mismatch)
    [55, 380.00, 1.90, 160, 2500.0],      # Flag=0 (Safe)
    [65, 420.00, 2.20, 175, 2700.0],      # Flag=0 (Safe)
    [75, 480.00, 2.70, 185, 2900.0],      # Flag=1 (Risky)
    [50, 360.00, 1.80, 165, 2550.0],      # Flag=0 (Safe)
    [90, 550.00, 3.00, 200, 3200.0],      # Flag=1 (Risky)
]).astype(float)

y_flag = np.array([0, 1, 0, 0, 1, 0, 0, 1, 0, 1])

# Train Random Forest Classifier
flag_model = RandomForestClassifier(
    n_estimators=10,
    max_depth=5,
    random_state=42
)
flag_model.fit(X_flag, y_flag)

# Save model
joblib.dump(flag_model, 'models/predict_flag_invoice.pkl')
print("✅ Flag model saved: models/predict_flag_invoice.pkl")

# ========================================================================
# Scaler (for feature scaling)
# ========================================================================
print("\n3️⃣  Creating Feature Scaler...")

scaler = StandardScaler()
scaler.fit(X_flag)

joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Scaler saved: models/scaler.pkl")

# ========================================================================
# Summary
# ========================================================================
print("\n" + "="*60)
print("✅ ALL DUMMY MODELS CREATED SUCCESSFULLY!")
print("="*60)
print("\nCreated files:")
print("  1. models/predict_freight_model.pkl")
print("  2. models/predict_flag_invoice.pkl")
print("  3. models/scaler.pkl")
print("\n⚠️  NOTE: These are dummy models trained on synthetic data.")
print("   Replace with real trained models from train.py when")
print("   you have actual training data (inventory.db)")
print("\n✅ You can now run: streamlit run app.py")
print("="*60)
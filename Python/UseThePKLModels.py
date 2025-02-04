import joblib
import numpy as np
import pandas as pd

# Load the trained models
linear_model = joblib.load("data/ML/LinearRegression_model.pkl")
rf_model = joblib.load("data/ML/RandomForest_model.pkl")
scaler = joblib.load("data/ML/scaler.pkl")  # If you saved a scaler

# Example input number (Change this to test different numbers)
test_number = 31

# Construct feature set based on the saved models' features
test_features = np.array([[test_number,  # Number itself
                           120,  # Example odd steps
                           60,   # Example even steps
                           15,   # Example odd descents
                           6.1,  # Max Log2
                           2.3,  # Min Log2
                           4.2,  # Mean Log2
                           1.1]]) # Variance Log2

# Define feature names exactly as used during training
feature_names = ["Number", "Odd Steps", "Even Steps", "Odd Descents", "Max Log2", "Min Log2", "Mean Log2", "Variance Log2"]

# Convert NumPy array to DataFrame before scaling
test_features_df = pd.DataFrame([test_features[0]], columns=feature_names)

# Scale the input features
test_features_scaled = scaler.transform(test_features_df)

# Make predictions
linear_prediction = linear_model.predict(test_features_scaled)
rf_prediction = rf_model.predict(test_features_scaled)

print(f"Predicted Collatz Steps (Linear Regression): {linear_prediction[0]}")
print(f"Predicted Collatz Steps (Random Forest): {rf_prediction[0]}")

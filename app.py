from flask import Flask, request, jsonify, render_template
import mlflow
import mlflow.keras
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Set the MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Load model from the MLflow Model Registry
model_name = "WaterQualityMLP"
model_version = 2  
model = mlflow.keras.load_model(f"models:/{model_name}/{model_version}")

# Load scaler used during training
scaler = joblib.load("artifacts/scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    data = request.form.to_dict()
    
    # Convert form data to DataFrame
    input_df = pd.DataFrame([data])
    
    # Ensure the input data contains all necessary features
    required_features = [
        "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
        "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"
    ]
    
    # Convert all feature values to numeric
    for feature in required_features:
        input_df[feature] = pd.to_numeric(input_df[feature], errors='coerce')
    
    # Preprocess input data using the same scaler
    X = scaler.transform(input_df)

    # Predict using the loaded model
    predictions_proba = model.predict(X)
    predictions = (predictions_proba > 0.5).astype(int).flatten()
    
    # Return predictions as JSON
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

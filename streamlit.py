import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle

# Load the saved model and associated objects
def load_model_and_objects():
    try:
        model = tf.keras.models.load_model('landslide_model.keras')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open('selector.pkl', 'rb') as f:
            selector = pickle.load(f)
        return model, scaler, feature_names, selector
    except Exception as e:
        st.error(f"Error loading model or objects: {e}")
        return None, None, None, None

# Prediction function
def predict_risks(model, scaler, selector, features, feature_names):
    """Predict multiple risks for input features"""
    # Convert input features to DataFrame with correct column names
    features_df = pd.DataFrame([features], columns=feature_names)
    # Scale the features
    features_scaled = scaler.transform(features_df)
    # Apply feature selection to reduce to 8 features
    features_selected = selector.transform(features_scaled)
    # Predict
    predictions = model.predict(features_selected, verbose=0)[0]
    
    return {
        'landslide_risk': predictions[0],
        'flood_risk': predictions[1],
        'erosion_risk': predictions[2],
        'vegetation_loss': predictions[3]
    }

# Streamlit UI
def main():
    st.title("Landslide Risk Prediction")
    st.subheader("Enter environmental parameters to predict risks")

    # Load model and objects
    model, scaler, feature_names, selector = load_model_and_objects()
    
    if model is None:
        st.stop()  # Stop execution if loading fails

    # Display expected feature names for transparency
    st.write("### Expected Features")
    st.write(f"Model trained on: {', '.join(feature_names)}")
    st.write(f"Expecting {len(feature_names)} features, selecting top {selector.k} for prediction")

    # Input fields for core features
    st.write("### Input Features")
    elevation_m = st.number_input("Elevation (meters)", min_value=0.0, max_value=5000.0, value=500.0, step=1.0)
    slope_deg = st.number_input("Slope (degrees)", min_value=0.0, max_value=90.0, value=25.0, step=1.0)
    rainfall_mm = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0, value=30.0, step=1.0)
    ndvi = st.number_input("NDVI (Normalized Difference Vegetation Index)", min_value=-1.0, max_value=1.0, value=0.4, step=0.1)
    soil_moisture = st.number_input("Soil Moisture (%)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
    temperature_c = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=25.0, step=1.0)
    landcover = st.number_input("Landcover (numeric code)", min_value=0, max_value=100, value=10, step=1)

    # Calculate interaction features
    slope_rainfall = slope_deg * rainfall_mm
    elevation_slope = elevation_m * slope_deg

    # Display calculated interaction features
    st.write("### Calculated Interaction Features")
    st.write(f"Slope × Rainfall: {slope_rainfall:.2f}")
    st.write(f"Elevation × Slope: {elevation_slope:.2f}")

    # Prepare features list in the order of feature_names
    input_features = [
        elevation_m,
        slope_deg,
        rainfall_mm,
        ndvi,
        soil_moisture,
        temperature_c,
        landcover,
        slope_rainfall,
        elevation_slope
    ]

    # Predict button
    if st.button("Predict Risks"):
        if len(input_features) != len(feature_names):
            st.error(f"Feature mismatch: Expected {len(feature_names)} features, got {len(input_features)}")
        else:
            try:
                risks = predict_risks(model, scaler, selector, input_features, feature_names)
                st.write("### Predicted Probabilities")
                st.success(f"Landslide Probability: {risks['landslide_risk']:.2%}")
                st.success(f"Flood Probability: {risks['flood_risk']:.2%}")
                st.success(f"Erosion Probability: {risks['erosion_risk']:.2%}")
                st.success(f"Vegetation Loss Probability: {risks['vegetation_loss']:.2%}")
            except Exception as e:
                st.error(f"Prediction error: {e}")

if __name__ == "__main__":
    main()
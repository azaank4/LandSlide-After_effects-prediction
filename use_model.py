import pandas as pd
import tensorflow as tf
import pickle

def load_model_and_predict():
    # Load the saved objects
    model = tf.keras.models.load_model('selector.pkl')
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    with open('selector.pkl', 'rb') as f:
        selector = pickle.load(f)
    
    # Example new data
    sample_features = [
        500,  # Elevation_m
        25,   # Slope_deg
        30,   # Rainfall_mm
        0.4,  # NDVI
        20,   # SoilMoisture
        298,  # Temperature_C
        10,   # Landcover
        25 * 30,  # Slope_Rainfall
        500 * 25  # Elevation_Slope
    ]
    
    # Predict risks
    features_df = pd.DataFrame([sample_features], columns=feature_names)
    features_scaled = scaler.transform(features_df)
    features_selected = selector.transform(features_scaled)
    predictions = model.predict(features_selected, verbose=0)[0]
    
    risks = {
        'landslide_risk': predictions[0],
        'flood_risk': predictions[1],
        'erosion_risk': predictions[2],
        'vegetation_loss': predictions[3]
    }
    
    print("Predicted risks:")
    print(f"Landslide probability: {risks['landslide_risk']:.2%}")
    print(f"Flood probability: {risks['flood_risk']:.2%}")
    print(f"Erosion probability: {risks['erosion_risk']:.2%}")
    print(f"Vegetation loss probability: {risks['vegetation_loss']:.2%}")

if __name__ == "__main__":
    load_model_and_predict()
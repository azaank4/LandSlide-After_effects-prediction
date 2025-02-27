import pandas as pd
import glob
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
import pickle

# Verify GPU availability and configure TensorFlow for CUDA
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU detected and configured for TensorFlow.")
else:
    print("No GPU detected. Running on CPU.")

def load_all_data(data_path='dataset/*.csv'):
    """Load and combine all CSV files from the dataset directory with error handling"""
    all_files = glob.glob(data_path)
    if not all_files:
        raise ValueError("No CSV files found in 'dataset' folder.")
    
    df_list = []
    for file in all_files:
        print(f"Loading file: {file}")
        try:
            if os.path.getsize(file) == 0:
                print(f"Skipping empty file: {file}")
                continue
            df_temp = pd.read_csv(file)
            if df_temp.empty:
                print(f"Skipping file with no data: {file}")
                continue
            print(f"Loaded {file} with {len(df_temp)} rows")
            df_list.append(df_temp)
        except pd.errors.EmptyDataError:
            print(f"Skipping file with no valid columns: {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    if not df_list:
        raise ValueError("No valid data loaded from any CSV files.")
    
    combined_df = pd.concat(df_list, ignore_index=True)
    print(f"Combined dataset size: {len(combined_df)} rows")
    return combined_df

def prepare_data(df):
    """Prepare features and create target variables"""
    print(f"Initial dataset size: {len(df)} rows")
    
    # Drop unused columns if they exist
    columns_to_drop = ['system:index', '.geo']
    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
    
    # Handle missing values
    df['Temperature_C'] = df['Temperature_C'].replace(0, df['Temperature_C'].median())
    df['SoilMoisture'] = df['SoilMoisture'].replace(0, df['SoilMoisture'].median())
    
    # Select core features
    features = [
        'Elevation_m', 'Slope_deg', 'Rainfall_mm', 'NDVI', 
        'SoilMoisture', 'Temperature_C', 'Landcover'
    ]
    
    # Create interaction features
    df['Slope_Rainfall'] = df['Slope_deg'] * df['Rainfall_mm']
    df['Elevation_Slope'] = df['Elevation_m'] * df['Slope_deg']
    features.extend(['Slope_Rainfall', 'Elevation_Slope'])
    
    # Define risk indicators
    df['landslide_risk'] = (
        ((df['Slope_deg'] > 30) & (df['Rainfall_mm'] > 20)) | 
        ((df['Slope_deg'] > 20) & (df['Rainfall_mm'] > 25) & (df['Elevation_m'] > 500)) |
        ((df['Slope_deg'] > 15) & (df['Rainfall_mm'] > 30) & (df['NDVI'] < 0.3))
    ).astype(int)
    
    df['flood_risk'] = ((df['Rainfall_mm'] > 25) & (df['WaterOccurrence'] > 0)).astype(int)
    df['erosion_risk'] = ((df['Slope_deg'] > 20) & (df['NDVI'] < 0.4)).astype(int)
    df['vegetation_loss'] = (df['NDVI'] < 0.3).astype(int)
    
    # One-hot encode Landcover if categorical
    if df['Landcover'].dtype in ['int64', 'float64']:
        pass
    else:
        df = pd.get_dummies(df, columns=['Landcover'], drop_first=True)
        features = [col for col in df.columns if col in features or col.startswith('Landcover_')]
    
    # Remove rows with missing values in required columns
    required_cols = features + ['landslide_risk', 'flood_risk', 'erosion_risk', 'vegetation_loss', 'WaterOccurrence']
    df = df.dropna(subset=[col for col in required_cols if col in df.columns])
    print(f"Dataset size after dropping NaNs: {len(df)} rows")
    
    # Remove duplicates
    df = df.drop_duplicates()
    print(f"Dataset size after removing duplicates: {len(df)} rows")
    
    X = df[features]
    y = df[['landslide_risk', 'flood_risk', 'erosion_risk', 'vegetation_loss']]
    
    return X, y, features

def apply_smote_per_target(X, y):
    """Apply SMOTE to each target individually and combine results"""
    smote = SMOTE(random_state=42)
    target_names = ['landslide_risk', 'flood_risk', 'erosion_risk', 'vegetation_loss']
    
    X_resampled = []
    y_resampled = []
    
    for i, target in enumerate(target_names):
        X_res, y_res = smote.fit_resample(X, y[target])
        y_full = np.zeros((len(y_res), 4))
        y_full[:, i] = y_res
        X_resampled.append(X_res)
        y_resampled.append(y_full)
        print(f"SMOTE for {target}: {len(y_res)} samples")
    
    X_combined = np.vstack(X_resampled)
    y_combined = np.vstack(y_resampled)
    
    X_unique, indices = np.unique(X_combined, axis=0, return_index=True)
    y_unique = np.zeros((len(X_unique), 4))
    
    for idx, x in enumerate(X_unique):
        mask = np.all(X_combined == x, axis=1)
        y_unique[idx] = np.mean(y_combined[mask], axis=0)
    
    print(f"Final resampled dataset size: {len(X_unique)} rows")
    return X_unique, y_unique

def train_model():
    """Train the multi-output neural network model and save it"""
    df = load_all_data()
    X, y, feature_names = prepare_data(df)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply SMOTE to each target
    X_res, y_res = apply_smote_per_target(X_scaled, y)
    
    # Feature selection
    selector = SelectKBest(f_classif, k=8)
    X_selected = selector.fit_transform(X_res, y_res[:, 0])  # Use landslide_risk
    
    # Build model
    model = tf.keras.Sequential([
        Dense(128, activation='relu', input_shape=(X_selected.shape[1],)),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(4, activation='sigmoid')  # 4 outputs
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train model
    history = model.fit(
        X_selected,
        y_res,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Save the model and associated objects
    model.save('landslide_model.keras')  # Use native Keras format
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    with open('selector.pkl', 'wb') as f:
        pickle.dump(selector, f)
    
    print("Model and associated objects saved successfully.")
    return model, scaler, feature_names, selector

def predict_risks(model, scaler, selector, features, feature_names):
    """Predict multiple risks for new data"""
    features_df = pd.DataFrame([features], columns=feature_names)
    features_scaled = scaler.transform(features_df)
    features_selected = selector.transform(features_scaled)
    predictions = model.predict(features_selected, verbose=0)[0]
    
    return {
        'landslide_risk': predictions[0],
        'flood_risk': predictions[1],
        'erosion_risk': predictions[2],
        'vegetation_loss': predictions[3]
    }

# Main execution
if __name__ == "__main__":
    try:
        model, scaler, feature_names, selector = train_model()
        
        # Example prediction
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
        
        risks = predict_risks(model, scaler, selector, sample_features, feature_names)
        print("Predicted risks (example):")
        print(f"Landslide probability: {risks['landslide_risk']:.2%}")
        print(f"Flood probability: {risks['flood_risk']:.2%}")
        print(f"Erosion probability: {risks['erosion_risk']:.2%}")
        print(f"Vegetation loss probability: {risks['vegetation_loss']:.2%}")
        
    except Exception as e:
        print(f"Error during execution: {e}")
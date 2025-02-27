# Landslide Prediction Model

This project implements a machine learning model to predict landslide risks and related environmental hazards using TensorFlow and Streamlit.

## Features
- Predicts landslide, flood, erosion, and vegetation loss risks
- Handles imbalanced data using SMOTE
- Includes feature engineering and selection
- Provides both CLI and web interface for predictions

## Requirements
- Python 3.9.9 (recommended)
- TensorFlow 2.10+
- Streamlit
- Pandas, NumPy, Scikit-learn, Imbalanced-learn

Install requirements:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
Run the training script:
```bash
python trainv2.py
```

This will:
1. Load and preprocess data from the dataset directory
2. Train a multi-output neural network model
3. Save the trained model and associated objects

### Making Predictions

#### CLI Interface
Run predictions from command line:
```bash
python use_model.py
```

#### Web Interface
Run the Streamlit app:
```bash
streamlit run streamlit.py
```

The web interface allows interactive input of environmental parameters and displays predicted risks.

## Model Details
The model uses the following features:
- Elevation (meters)
- Slope (degrees)
- Rainfall (mm)
- NDVI (Normalized Difference Vegetation Index)
- Soil Moisture (%)
- Temperature (°C)
- Landcover (numeric code)
- Slope × Rainfall (interaction feature)
- Elevation × Slope (interaction feature)

## Dataset
The dataset contains environmental data from Kerala, India from 2020-2024, organized by month and batch.

## File Structure
```
.
├── dataset/                  # Contains CSV data files
├── landslide_model.keras     # Trained model
├── scaler.pkl               # Feature scaler
├── feature_names.pkl        # Feature names
├── selector.pkl             # Feature selector
├── trainv2.py               # Training script
├── use_model.py             # CLI prediction script
└── streamlit.py             # Web interface
```

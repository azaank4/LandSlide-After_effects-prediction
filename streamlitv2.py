import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the model class that matches the saved model
class LandslideMultiTargetModel:
    def __init__(self):
        self.preprocessor = None
        self.classifiers = {}
        self.regressors = {}
        self.target_encoders = {}
    
    def predict(self, X):
        predictions = {}
        
        # Process input data if preprocessor exists
        if self.preprocessor:
            X = self.preprocessor.transform(X)
            
        # Make predictions for classifiers
        for target, classifier in self.classifiers.items():
            if classifier is not None:
                pred = classifier.predict(X)
                if target in self.target_encoders:
                    pred = self.target_encoders[target].inverse_transform(pred)
                predictions[target] = pred[0] if len(pred) == 1 else pred
                
        # Make predictions for regressors
        for target, regressor in self.regressors.items():
            if regressor is not None:
                pred = regressor.predict(X)
                predictions[target] = pred[0] if len(pred) == 1 else pred
                
        return predictions

# Load the multi-target model with caching and error handling
@st.cache_resource
def load_model():
    try:
        model_path = os.path.join('models', 'landslide_multi_target_model.joblib')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Define mappings for categorical predictions
TYPE_OF_SL_MAP = {
    'Df': 'Debris Flow',
    'DF': 'Debris Flow',  # Added uppercase variant
    'SS': 'Surficial Slide',
    'RF': 'Rock Fall'
}

ROAD_IMPACT_MAP = {
    'C': 'Land Covers the Road',
    'D': 'Landslide Destroys the Road',
    'N': 'No Impact'
}

IMPACT_AGR_MAP = {
    'N': 'No Impact',
    'BRO': 'Bare Rock',
    'BRG': 'Bare Rock with Isolated Grass',
    'BRS': 'Bare Rock with Isolated Shrubs',
    'BRF': 'Bare Rock with Isolated Forests',
    'BSL': 'Bare Farmland',
    'BSO': 'Bare Soil',
    'BSG': 'Bare Soil with Isolated Grass',
    'BSS': 'Bare Soil with Isolated Shrubs',
    'BSF': 'Bare Soil with Isolated Trees',
    'GNA': 'Natural Grassland',
    'GMC': 'Meadows (Cultivated Grassland)',
    'SNA': 'Natural Shrub Land',
    'SPL': 'Shrub Plantation',
    'FNO': 'Open Natural Forest',
    'FDN': 'Dense Natural Forest',
    'FCP': 'Forest Plantation',
    'FMP': 'Mixed Forest Plantation',
    'TEA': 'Tea Plantation',
    'RUB': 'Rubber Plantation',
    'CSB': 'Bare Cut Slopes',
    'CSV': 'Vegetated Cut Slopes',
    'BUI': 'Buildings',
    'ROA': 'Road',
    'QUU': 'Quarry in Use'
}

def display_predictions(predictions):
    """Display predictions in a user-friendly format."""
    st.header('Prediction Results')

    # Categorical Predictions
    st.subheader('Categorical Predictions')
    cat_cols = ['Type_of_sl', 'Road_impac', 'Impact_Agr']
    cat_results = {k: predictions[k] for k in cat_cols if k in predictions}

    col1, col2, col3 = st.columns(3)
    for i, (key, value) in enumerate(cat_results.items()):
        if key == 'Type_of_sl':
            label = 'Type of Landslide'
            # Convert value to uppercase for consistent mapping
            value = str(value).upper()  # Convert to string and uppercase
            value = TYPE_OF_SL_MAP.get(value) or TYPE_OF_SL_MAP.get('DF' if value == 'DF' else value) or value
        elif key == 'Road_impac':
            label = 'Road Impact'
            value = ROAD_IMPACT_MAP.get(value, value)
        elif key == 'Impact_Agr':
            label = 'Impact on Agriculture'
            value = IMPACT_AGR_MAP.get(value, value)
        else:
            label = key.replace('_', ' ').title()
        [col1, col2, col3][i % 3].metric(label, value)

    # Numerical Predictions
    st.subheader('Numerical Predictions')
    num_cols = ['Length', 'Width', 'Area', 'Building_I']
    num_results = {k: predictions[k] for k in num_cols if k in predictions}

    col1, col2, col3 = st.columns(3)
    for i, (key, value) in enumerate(num_results.items()):
        label = key.replace('_', ' ').title()
        [col1, col2, col3][i % 3].metric(label, f"{value:.2f}")

def main():
    st.set_page_config(page_title="Landslide Prediction System", layout="wide")
    st.title('🌍 Landslide Multi-Target Prediction System')
    st.markdown("""
        This application predicts various landslide characteristics based on geographical and environmental factors.
        Use the form below to input data or upload a CSV file for batch predictions.
    """)

    model = load_model()

    # Tabs for input methods
    tab1, tab2 = st.tabs(["📋 Form Input", "📁 File Upload"])

    with tab1:
        st.header('Input Data')
        with st.form("input_form"):
            col1, col2 = st.columns(2)

            with col1:
                district = st.selectbox('District', 
                                        ['Idukki', 'Wayanad', 'Kozhikode', 'Malappuram', 'Palakkad'],
                                        help="Select the district where the landslide might occur")
                lu_2018 = st.selectbox('Land Use 2018', 
                                      ['Forest', 'Agriculture', 'Built-up', 'Barren', 'Water'],
                                      help="Land use classification as of 2018")

            with col2:
                elevation = st.number_input('Elevation (m)', 
                                          min_value=0.0, max_value=3000.0, value=1000.0,
                                          help="Elevation in meters above sea level")
                soil_type = st.selectbox('Soil Type', 
                                        ['Laterite', 'Forest', 'Alluvial', 'Rocky', 'Clay'],
                                        help="Type of soil in the area")
                precipitation = st.number_input('Precipitation 2018 (mm)', 
                                             min_value=0.0, max_value=5000.0, value=2000.0,
                                             help="Annual precipitation in millimeters for 2018")

            submitted = st.form_submit_button("Predict")
            if submitted:
                input_data = pd.DataFrame({
                    'District': [district],
                    'LU_2018': [lu_2018],
                    'elevation_m': [elevation],
                    'soil_type': [soil_type],
                    'precip_2018_mm': [precipitation]
                })

                try:
                    predictions = model.predict(input_data)
                    display_predictions(predictions)
                except Exception as e:
                    st.error(f"Error making prediction: {e}")

    with tab2:
        st.header("Upload a CSV File")
        st.info("Required columns: 'District', 'LU_2018', 'elevation_m', 'soil_type', 'precip_2018_mm'")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

        if uploaded_file:
            try:
                test_data = pd.read_csv(uploaded_file)
                st.dataframe(test_data.head())

                if st.button("Predict on Uploaded Data"):
                    try:
                        predictions = model.predict(test_data)
                        st.header("Batch Prediction Results")
                        st.dataframe(predictions)

                        csv = predictions.to_csv(index=False)
                        st.download_button(
                            label="Download Predictions as CSV",
                            data=csv,
                            file_name="predictions.csv",
                            mime="text/csv"
                        )
                    except Exception as e:
                        st.error(f"Error making batch predictions: {e}")
            except Exception as e:
                st.error(f"Error processing file: {e}")

    st.sidebar.header('ℹ️ About the Model')
    st.sidebar.info("""
        This multi-target prediction model was trained on landslide data from Kerala (2018).
        It predicts multiple characteristics of landslides based on geographical and environmental factors.
    """)

if __name__ == '__main__':
    main()
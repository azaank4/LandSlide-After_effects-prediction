import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Define a flexible model class that can work with however your model was structured
class LandslideMultiTargetModel:
    def __init__(self):
        # We'll define the predict method only, since that's what's being called
        pass
        
    def predict(self, X):
        # This is a placeholder method that will be replaced by the actual method from the loaded model
        # The internal implementation doesn't matter as it will be overridden when loaded
        pass

# Load the multi-target model with debugging
@st.cache_resource
def load_model():
    model = joblib.load(os.path.join('models', 'landslide_multi_target_model.joblib'))
    # Display model structure for debugging if needed
    # st.write("Model type:", type(model))
    # st.write("Model attributes:", dir(model))
    return model

def main():
    st.title('Landslide Multi-Target Prediction System')
    st.write('This app predicts various landslide characteristics based on geographical and environmental factors.')
    
    try:
        model = load_model()
        
        # Create tabs for input methods
        tab1, tab2 = st.tabs(["Form Input", "File Upload"])
        
        with tab1:
            st.header('Input Data')
            
            # Create form inputs for features
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
            
            if st.button('Predict'):
                # Create input dataframe
                input_data = pd.DataFrame({
                    'District': [district],
                    'LU_2018': [lu_2018],
                    'elevation_m': [elevation],
                    'soil_type': [soil_type],
                    'precip_2018_mm': [precipitation]
                })
                
                # Make prediction with error handling
                try:
                    prediction = {}
                    
                    # Preprocess the input data first
                    if hasattr(model, 'preprocessor'):
                        processed_data = model.preprocessor.transform(input_data)
                    else:
                        processed_data = input_data
                    
                    # Get predictions for all targets using the model's attributes
                    if hasattr(model, 'classifiers') and hasattr(model, 'regressors'):
                        # Handle categorical predictions
                        for target, classifier in model.classifiers.items():
                            if classifier is not None:
                                pred = classifier.predict(processed_data)
                                # Inverse transform if target encoder exists
                                if hasattr(model, 'target_encoders') and target in model.target_encoders:
                                    pred = model.target_encoders[target].inverse_transform(pred)
                                prediction[target] = pred[0]
                        
                        # Handle numerical predictions
                        for target, regressor in model.regressors.items():
                            if regressor is not None:
                                pred = regressor.predict(processed_data)
                                prediction[target] = pred[0]
                    else:
                        # Fallback to direct prediction
                        pred = model.predict(processed_data)
                        if isinstance(pred, dict):
                            prediction = {k: v[0] if isinstance(v, np.ndarray) else v 
                                       for k, v in pred.items()}
                        else:
                            raise ValueError("Unexpected prediction format")

                    if not prediction:
                        raise ValueError("No predictions were generated")
                    
                    # Display all predictions
                    st.header('Prediction Results')
                    
                    # Categorical predictions
                    st.subheader('Categorical Predictions')
                    cat_cols = ['Type_of_sl', 'Road_impac', 'Impact_Agr']
                    cat_results = {k: prediction[k] for k in cat_cols if k in prediction}
                    
                    # Maps for categorical values
                    type_of_sl_map = {
                        'Df': 'Debris Flow',
                        'SS': 'Surficial Slide',
                        'RF': 'Rock Fall'
                    }

                    road_impac_map = {
                        'C': 'Land Covers the Road',
                        'D': 'Landslide Destroys the Road',
                        'N': 'No Impact'
                    }

                    impact_agr_map = {
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

                    # Display categorical predictions using metrics
                    cat_cols_ui = st.columns(len(cat_results))
                    for i, (key, value) in enumerate(cat_results.items()):
                        if key == 'Type_of_sl':
                            formatted_key = 'Type of Landslide'  # Change the label
                            value = type_of_sl_map.get(value, value)  # Map the value to its description
                        elif key == 'Road_impac':
                            formatted_key = 'Road Impact'  # Change the label
                            value = road_impac_map.get(value, value)  # Map the value to its description
                        elif key == 'Impact_Agr':
                            formatted_key = 'Impact on Agriculture'  # Change the label
                            value = impact_agr_map.get(value, value)  # Map the value to its description
                        else:
                            formatted_key = key.replace('_', ' ').title()
                        cat_cols_ui[i].metric(formatted_key, value)
                    
                    # Numerical predictions
                    st.subheader('Numerical Predictions')
                    num_cols = ['Length', 'Width', 'Area', 'Building_I']
                    num_results = {k: prediction[k] for k in num_cols if k in prediction}
                    
                    # Display numerical predictions using metrics
                    num_cols_ui = st.columns(len(num_results))
                    for i, (key, value) in enumerate(num_results.items()):
                        formatted_key = key.replace('_', ' ').title()
                        num_cols_ui[i].metric(formatted_key, f"{value:.2f}")
                    
                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    st.error("Model debug info: " + str(type(model)))
                    st.error("Model attributes: " + str(dir(model)))
        
        with tab2:
            st.write("Upload a CSV file with the required features:")
            st.info("Required columns: 'District', 'LU_2018', 'elevation_m', 'soil_type', 'precip_2018_mm'")
            
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            
            if uploaded_file is not None:
                try:
                    test_data = pd.read_csv(uploaded_file)
                    
                    # Check for required columns
                    required_cols = ['District', 'LU_2018', 'elevation_m', 'soil_type', 'precip_2018_mm']
                    missing_cols = set(required_cols) - set(test_data.columns)
                    
                    if missing_cols:
                        st.error(f"Missing required columns: {missing_cols}")
                    else:
                        st.dataframe(test_data.head())
                        
                        if st.button('Predict on Uploaded Data'):
                            # Make predictions with error handling
                            try:
                                predictions = model.predict(test_data)
                                
                                # Convert to DataFrame for display
                                if isinstance(predictions, dict):
                                    results = {}
                                    for k, v in predictions.items():
                                        if isinstance(v, np.ndarray):
                                            results[k] = v
                                        else:
                                            results[k] = [v]
                                    
                                    results_df = pd.DataFrame(results)
                                else:
                                    # If predictions is already a DataFrame
                                    results_df = predictions if isinstance(predictions, pd.DataFrame) else pd.DataFrame(predictions)
                                
                                st.header("Prediction Results")
                                st.dataframe(results_df)
                                
                                # Show visualizations for batch predictions
                                st.subheader('Distribution of Categorical Predictions')
                                
                                # Plot categorical predictions
                                cat_cols = [col for col in ['Type_of_sl', 'Road_impac', 'Impact_Agr'] 
                                          if col in results_df.columns]
                                if cat_cols:
                                    fig, axes = plt.subplots(1, len(cat_cols), figsize=(15, 5))
                                    if len(cat_cols) == 1:
                                        axes = [axes]  # Make it iterable if only one subplot
                                        
                                    for i, col in enumerate(cat_cols):
                                        results_df[col].value_counts().plot(kind='bar', ax=axes[i])
                                        axes[i].set_title(f'{col} Distribution')
                                        axes[i].set_ylabel('Count')
                                        plt.xticks(rotation=45)
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                
                                # Plot numerical predictions
                                st.subheader('Distribution of Numerical Predictions')
                                num_cols = [col for col in ['Length', 'Width', 'Area', 'Building_I'] 
                                           if col in results_df.columns]
                                if num_cols:
                                    fig, axes = plt.subplots(1, len(num_cols), figsize=(15, 5))
                                    if len(num_cols) == 1:
                                        axes = [axes]  # Make it iterable if only one subplot
                                        
                                    for i, col in enumerate(num_cols):
                                        axes[i].hist(results_df[col], bins=10)
                                        axes[i].set_title(f'{col} Distribution')
                                        axes[i].set_xlabel(col)
                                        axes[i].set_ylabel('Frequency')
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                
                                # Add download button for predictions
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="Download predictions as CSV",
                                    data=csv,
                                    file_name="landslide_predictions.csv",
                                    mime="text/csv",
                                )
                            except Exception as e:
                                st.error(f"Error making batch predictions: {e}")
                                st.error("Model debug info: " + str(type(model)))
                                st.error("Model attributes: " + str(dir(model)))
                except Exception as e:
                    st.error(f"Error processing file: {e}")
        
        # Add information about the model
        st.sidebar.header('Model Information')
        st.sidebar.info(
            "This multi-target prediction model was trained on landslide data from Kerala (2018). "
            "It predicts multiple characteristics of landslides based on geographical and environmental factors."
        )
        
        # Feature importance visualization
        if os.path.exists('plots'):
            st.sidebar.header('Feature Importance')
            importance_plots = [f for f in os.listdir('plots') if f.startswith('feature_importance_')]
            
            if importance_plots:
                selected_plot = st.sidebar.selectbox('Select Target Variable', 
                                                    [p.replace('feature_importance_', '').replace('.png', '') 
                                                     for p in importance_plots])
                
                plot_path = f'plots/feature_importance_{selected_plot}.png'
                if os.path.exists(plot_path):
                    st.sidebar.image(plot_path, caption=f'Feature Importance for {selected_plot}')
    
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please make sure the model file exists in the 'models' directory.")
        # Add extended debugging info
        st.error(f"Error details: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == '__main__':
    main()
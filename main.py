import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import classification_report, mean_squared_error

# Load dataset
data = pd.read_csv('kerala_2018_dataset.csv')

# --- Data Cleaning ---
# Drop Reclass_Sl if constant
if 'Reclass_Sl' in data.columns and data['Reclass_Sl'].nunique() == 1:
    data.drop('Reclass_Sl', axis=1, inplace=True)

# Encode target variable (Type_of_sl)
label_encoder = LabelEncoder()
y_class = label_encoder.fit_transform(data['Type_of_sl'])

# Separate targets for regression
y_reg_length = data['Length']
y_reg_width = data['Width']
y_reg_area = data['Area']

# Features (X) - dropping the target variables
X = data.drop(['Type_of_sl', 'Length', 'Width', 'Area'], axis=1)

# --- Feature Preparation ---
# Define categorical and numerical columns AFTER dropping targets
# Make sure these columns actually exist in X
categorical_cols = [col for col in ['District', 'LU_2010', 'LU_2018', 'soil_type', 'Road_impac', 'Impact_Agr'] 
                   if col in X.columns]
numerical_cols = [col for col in ['Building_I', 'RASTERVALU', 'longitude', 'latitude', 
                                 'elevation_m', 'precip_2018_mm']
                 if col in X.columns]

# Print column names to verify
print("X columns:", X.columns.tolist())
print("Categorical columns being used:", categorical_cols)
print("Numerical columns being used:", numerical_cols)

# Preprocess the data with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numerical_cols)
    ],
    remainder='drop'
)

# --- Split Data ---
X_train, X_test, y_train_class, y_test_class, \
y_train_length, y_test_length, \
y_train_width, y_test_width, \
y_train_area, y_test_area = train_test_split(
    X, y_class, y_reg_length, y_reg_width, y_reg_area,
    test_size=0.2, random_state=42
)

# Apply preprocessing
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# --- Scale Data ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)

# --- Apply SMOTE for Classification ---
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train_class)

print("Original class distribution:")
print(pd.Series(y_train_class).value_counts())
print("\nResampled class distribution:")
print(pd.Series(y_train_resampled).value_counts())

# --- Train Classifier ---
classifier = RandomForestClassifier(random_state=42)
classifier.fit(X_train_resampled, y_train_resampled)

# Evaluate Classifier
y_pred_class = classifier.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test_class, y_pred_class, target_names=label_encoder.classes_))

# --- Train Regressors ---
regressor_length = GradientBoostingRegressor(random_state=42)
regressor_width = GradientBoostingRegressor(random_state=42)
regressor_area = GradientBoostingRegressor(random_state=42)

# Fit regressors
regressor_length.fit(X_train_scaled, y_train_length)
regressor_width.fit(X_train_scaled, y_train_width)
regressor_area.fit(X_train_scaled, y_train_area)

# Evaluate Regressors
y_pred_length = regressor_length.predict(X_test_scaled)
y_pred_width = regressor_width.predict(X_test_scaled)
y_pred_area = regressor_area.predict(X_test_scaled)

print("\nRegression Metrics:")
print("Length RMSE:", np.sqrt(mean_squared_error(y_test_length, y_pred_length)))
print("Width RMSE:", np.sqrt(mean_squared_error(y_test_width, y_pred_width)))
print("Area RMSE:", np.sqrt(mean_squared_error(y_test_area, y_pred_area)))

# --- Hybrid Model Prediction ---
def hybrid_predict(X_new):
    """
    Predict landslide type and dimensions using the hybrid model.
    """
    # Make sure X_new has the same columns as X
    X_new_aligned = X_new.reindex(columns=X.columns, fill_value=0)
    
    # Preprocess new data
    X_new_processed = preprocessor.transform(X_new_aligned)
    X_new_scaled = scaler.transform(X_new_processed)
    
    # Predict landslide type
    landslide_type = classifier.predict(X_new_scaled)
    landslide_type_decoded = label_encoder.inverse_transform(landslide_type)
    
    # Predict dimensions
    length_pred = regressor_length.predict(X_new_scaled)
    width_pred = regressor_width.predict(X_new_scaled)
    area_pred = regressor_area.predict(X_new_scaled)
    
    return landslide_type_decoded, length_pred, width_pred, area_pred

# Example usage of hybrid_predict
example_data = X_test.iloc[:5]  # Use first 5 rows of test data as an example
predicted_type, predicted_length, predicted_width, predicted_area = hybrid_predict(example_data)

print("\nHybrid Model Predictions:")
for i in range(len(predicted_type)):
    print(f"Sample {i+1}:")
    print(f"  Type: {predicted_type[i]}")
    print(f"  Length: {predicted_length[i]:.2f}")
    print(f"  Width: {predicted_width[i]:.2f}")
    print(f"  Area: {predicted_area[i]:.2f}")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, learning_curve
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_squared_error, confusion_matrix, r2_score
from sklearn.inspection import permutation_importance
import joblib
import os

# Create directory for saving models and plots
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Load dataset
data = pd.read_csv('kerala_2018_dataset.csv')

# --- Define features and targets ---
# Features (inputs)
feature_cols = ['District', 'LU_2018', 'elevation_m', 'soil_type', 'precip_2018_mm']

# Target variables (outputs to predict)
categorical_targets = ['Type_of_sl', 'Road_impac', 'Impact_Agr']
numerical_targets = ['Length', 'Width', 'Area', 'Building_I']

# Verify columns exist
print("Features in dataset:", [col for col in feature_cols if col in data.columns])
print("Categorical targets in dataset:", [col for col in categorical_targets if col in data.columns])
print("Numerical targets in dataset:", [col for col in numerical_targets if col in data.columns])

# --- Prepare features (X) ---
X = data[feature_cols].copy()

# Define categorical and numerical feature columns
categorical_features = ['District', 'LU_2018', 'soil_type']
numerical_features = ['elevation_m', 'precip_2018_mm']

# --- Prepare targets (y) ---
# Encode categorical targets
target_encoders = {}
y_categorical_encoded = {}

for target in categorical_targets:
    if target in data.columns:
        encoder = LabelEncoder()
        y_categorical_encoded[target] = encoder.fit_transform(data[target])
        target_encoders[target] = encoder
        print(f"Classes for {target}:", encoder.classes_)

# Get numerical targets
y_numerical = {}
for target in numerical_targets:
    if target in data.columns:
        y_numerical[target] = data[target].values

# --- Preprocessing for features ---
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numerical_features)
    ],
    remainder='drop'
)

# --- Split Data ---
# Initialize dictionaries to store splits
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

y_train_cat = {}
y_test_cat = {}
for target in categorical_targets:
    if target in data.columns:
        y_train_cat[target], y_test_cat[target] = train_test_split(
            y_categorical_encoded[target], test_size=0.2, random_state=42
        )

y_train_num = {}
y_test_num = {}
for target in numerical_targets:
    if target in data.columns:
        y_train_num[target], y_test_num[target] = train_test_split(
            y_numerical[target], test_size=0.2, random_state=42
        )

# Apply preprocessing to features
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Save preprocessor
joblib.dump(preprocessor, 'models/preprocessor.joblib')
joblib.dump(target_encoders, 'models/target_encoders.joblib')

# --- Train models for categorical targets ---
classifiers = {}
for target in categorical_targets:
    if target in data.columns:
        print(f"\nTraining classifier for: {target}")
        
        # Apply SMOTE for imbalanced classes
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train_cat[target])
        
        # Train classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train_resampled, y_train_resampled)
        classifiers[target] = clf
        
        # Evaluate classifier
        y_pred = clf.predict(X_test_processed)
        
        # Print classification report
        print(f"Classification Report for {target}:")
        print(classification_report(y_test_cat[target], y_pred, 
                                    target_names=target_encoders[target].classes_))
        
        # Plot confusion matrix
        cm = confusion_matrix(y_test_cat[target], y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=target_encoders[target].classes_, 
                    yticklabels=target_encoders[target].classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix for {target}')
        plt.tight_layout()
        plt.savefig(f'plots/confusion_matrix_{target}.png')
        plt.close()
        
        # Save model
        joblib.dump(clf, f'models/classifier_{target}.joblib')

# --- Train models for numerical targets ---
regressors = {}
for target in numerical_targets:
    if target in data.columns:
        print(f"\nTraining regressor for: {target}")
        
        # Train regressor
        reg = RandomForestRegressor(n_estimators=100, random_state=42)
        reg.fit(X_train_processed, y_train_num[target])
        regressors[target] = reg
        
        # Evaluate regressor
        y_pred = reg.predict(X_test_processed)
        rmse = np.sqrt(mean_squared_error(y_test_num[target], y_pred))
        r2 = r2_score(y_test_num[target], y_pred)
        
        print(f"RMSE for {target}: {rmse:.4f}")
        print(f"R² score for {target}: {r2:.4f}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test_num[target], y_pred, alpha=0.5)
        plt.plot([min(y_test_num[target]), max(y_test_num[target])], 
                 [min(y_test_num[target]), max(y_test_num[target])], 
                 'k--', lw=2)
        plt.xlabel(f'Actual {target}')
        plt.ylabel(f'Predicted {target}')
        plt.title(f'{target} Predictions (RMSE: {rmse:.4f}, R²: {r2:.4f})')
        plt.tight_layout()
        plt.savefig(f'plots/regression_predictions_{target}.png')
        plt.close()
        
        # Save model
        joblib.dump(reg, f'models/regressor_{target}.joblib')

# --- Feature importance analysis ---
# Get feature names after transformation
cat_encoder = preprocessor.named_transformers_['cat']
cat_feature_names = []
if hasattr(cat_encoder, 'get_feature_names_out'):
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
else:
    # For older sklearn versions
    categories = cat_encoder.categories_
    for i, feature in enumerate(categorical_features):
        for j in range(len(categories[i]) - 1):  # -1 because drop='first'
            cat_feature_names.append(f"{feature}_{categories[i][j+1]}")

feature_names = list(cat_feature_names) + numerical_features

# Plot feature importance for each target
for target in categorical_targets + numerical_targets:
    if target in data.columns:
        # Get model
        model = classifiers.get(target) if target in categorical_targets else regressors.get(target)
        
        if model is not None:
            # Get feature importance
            importances = model.feature_importances_
            
            # Ensure feature_names matches length of importances
            if len(feature_names) == len(importances):
                # Sort features by importance
                indices = np.argsort(importances)[-15:]  # Top 15 features
                
                plt.figure(figsize=(12, 8))
                plt.barh(range(len(indices)), importances[indices])
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                plt.xlabel('Feature Importance')
                plt.title(f'Top 15 Feature Importances for {target}')
                plt.tight_layout()
                plt.savefig(f'plots/feature_importance_{target}.png')
                plt.close()
            else:
                print(f"Feature names length ({len(feature_names)}) doesn't match " +
                      f"importance length ({len(importances)}) for {target}")

# --- Create a unified prediction class ---
class LandslideMultiTargetModel:
    def __init__(self, preprocessor, target_encoders, classifiers, regressors, feature_columns):
        """
        Initialize the multi-target model with all its components.

        Args:
            preprocessor: The ColumnTransformer object for feature preprocessing
            target_encoders: Dict mapping target name to its label encoder
            classifiers: Dict mapping target name to its classifier
            regressors: Dict mapping target name to its regressor
            feature_columns: List of feature column names
        """
        self.preprocessor = preprocessor
        self.target_encoders = target_encoders
        self.classifiers = classifiers
        self.regressors = regressors
        self.feature_columns = feature_columns
        
    def predict(self, X_new):
        """
        Make predictions for all targets using the input features.
        
        Args:
            X_new: DataFrame with required input features
            
        Returns:
            dict: Dictionary containing predictions for all targets
        """
        # Make sure X_new has the required columns
        missing_cols = set(self.feature_columns) - set(X_new.columns)
        if missing_cols:
            raise ValueError(f"Missing required feature columns: {missing_cols}")
        
        # Align columns with expected order
        X_new_aligned = X_new[self.feature_columns]
        
        # Preprocess input features
        X_new_processed = self.preprocessor.transform(X_new_aligned)
        
        # Initialize results container
        all_predictions = []
        for i in range(len(X_new)):
            row_predictions = {}
            
            # Get a single sample for prediction
            X_sample = X_new_processed[i:i+1]
            
            # Make predictions for categorical targets
            for target, classifier in self.classifiers.items():
                pred_encoded = classifier.predict(X_sample)[0]
                pred_decoded = self.target_encoders[target].inverse_transform([pred_encoded])[0]
                row_predictions[target] = pred_decoded
            
            # Make predictions for numerical targets
            for target, regressor in self.regressors.items():
                pred_value = regressor.predict(X_sample)[0]
                row_predictions[target] = pred_value
                
            all_predictions.append(row_predictions)
        
        return all_predictions if len(all_predictions) > 1 else all_predictions[0]

# Create and save the multi-target model
multi_target_model = LandslideMultiTargetModel(
    preprocessor=preprocessor,
    target_encoders=target_encoders,
    classifiers=classifiers,
    regressors=regressors,
    feature_columns=feature_cols
)

joblib.dump(multi_target_model, 'models/landslide_multi_target_model.joblib')

# --- Example prediction ---
print("\nExample prediction:")
sample_data = X_test.iloc[:3]  # Use first 3 rows of test data

# Make predictions
predictions = multi_target_model.predict(sample_data)

# Display predictions
if isinstance(predictions, list):
    for i, pred in enumerate(predictions):
        print(f"\nSample {i+1}:")
        for target, value in pred.items():
            print(f"  {target}: {value}")
else:
    for target, value in predictions.items():
        print(f"  {target}: {value}")

print("\nModel training and evaluation completed.")
print("All models and preprocessors have been saved in the 'models' directory.")
print("All evaluation plots have been saved in the 'plots' directory.")
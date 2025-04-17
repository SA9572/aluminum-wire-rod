import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import os
import joblib

def load_data(data_dir):
    """Load preprocessed test data and scaler.."""
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))
    scaler = joblib.load(os.path.join(data_dir, 'scaler.pkl'))  # Load scaler (for features, not used here)
    
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)
    print("y_test sample:\n", y_test.head())
    
    return X_test, y_test, scaler

def classify_quality(y_data, tensile_threshold=400, elongation_threshold=25):
    """Classify quality based on thresholds using if-else logic."""
    classifications = []
    
    for i in range(len(y_data)):
        tensile_strength = y_data[i, 1]  # Tensile Strength (second column)
        elongation = y_data[i, 2]        # Elongation (third column)
        
        if tensile_strength > tensile_threshold and elongation > elongation_threshold:
            classifications.append(1)  # Good quality
        else:
            classifications.append(0)  # Poor quality
    
    return np.array(classifications)

def suggest_improvements(y_data, tensile_threshold=400, elongation_threshold=25):
    """Provide rule-based suggestions based on material science principles."""
    suggestions = []
    
    for i in range(len(y_data)):
        proof_stress = y_data[i, 0]      # 0.2% Proof Stress (MPa)
        tensile_strength = y_data[i, 1]  # Tensile Strength (MPa)
        elongation = y_data[i, 2]        # Elongation (%)
        
        if tensile_strength <= tensile_threshold and elongation <= elongation_threshold:
            suggestions.append("Increase annealing time and temperature to improve both tensile strength and ductility.")
        elif tensile_strength <= tensile_threshold:
            suggestions.append("Increase annealing time or adjust alloy composition (e.g., add strengthening elements like Si or Mn).")
        elif elongation <= elongation_threshold:
            suggestions.append("Reduce cold working or increase annealing temperature to enhance ductility.")
        else:
            suggestions.append("Quality meets standards; no improvements needed.")
    
    return suggestions

def evaluate_classification(y_true, y_pred, model_name):
    """Evaluate classification accuracy and provide suggestions."""
    true_class = classify_quality(y_true)
    pred_class = classify_quality(y_pred)
    
    # Debug: Print class distributions
    print(f"{model_name} - True class distribution: {np.bincount(true_class)}")
    print(f"{model_name} - Predicted class distribution: {np.bincount(pred_class)}")
    
    accuracy = accuracy_score(true_class, pred_class)
    
    # Handle single-class case
    unique_classes = np.unique(np.concatenate([true_class, pred_class]))
    if len(unique_classes) < 2:
        print(f"\n{model_name} Classification Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Warning: Only one class present in true or predicted labels.")
        print(f"Class present: {'Poor' if unique_classes[0] == 1 else 'Good'}")
    else:
        report = classification_report(true_class, pred_class, target_names=['Good', 'Poor'], labels=[0, 1])
        print(f"\n{model_name} Classification Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", report)
    
    # Generate suggestions for predicted values
    suggestions = suggest_improvements(y_pred)
    print(f"\n{model_name} Improvement Suggestions (first 5 samples):")
    for i, suggestion in enumerate(suggestions[:5]):
        print(f"Sample {i}: {suggestion}")

def test_suggestions():
    """Test the suggestion system with sample inputs."""
    sample_inputs = np.array([
        [250, 380, 20],  # Low tensile strength, low elongation
        [300, 450, 30],  # Good tensile strength, good elongation
        [200, 350, 28],  # Low tensile strength, good elongation
        [280, 420, 22],  # Good tensile strength, low elongation
        [150, 300, 15]   # Very low tensile strength and elongation
    ])
    print("\nTesting Suggestions with Sample Inputs:")
    suggestions = suggest_improvements(sample_inputs)
    for i, (input_data, suggestion) in enumerate(zip(sample_inputs, suggestions)):
        print(f"Sample {i} - Proof: {input_data[0]}, Tensile: {input_data[1]}, Elongation: {input_data[2]}")
        print(f"Suggestion: {suggestion}\n")

if __name__ == "__main__":
    # Define paths
    data_dir = "C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/data"
    models_dir = "C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/models"

    # Load test data and scaler
    X_test, y_test, scaler = load_data(data_dir)

    # Load trained models
    lr_model = joblib.load(os.path.join(models_dir, 'linear_regression.pkl'))
    rf_model = joblib.load(os.path.join(models_dir, 'random_forest_tuned.pkl'))
    xgb_model = joblib.load(os.path.join(models_dir, 'xgboost_tuned.pkl'))

    # Make predictions
    lr_pred = lr_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)

    # Define thresholds (adjusted for more variation)
    tensile_threshold = 400  # MPa
    elongation_threshold = 25  # %

    # Convert y_test to numpy array with correct column names
    y_test_array = y_test[[' 0.2% Proof Stress (MPa)', ' Tensile Strength (MPa)', ' Elongation (%)']].values

    # Evaluate classification and suggestions for each model
    evaluate_classification(y_test_array, lr_pred, "Linear Regression")
    evaluate_classification(y_test_array, rf_pred, "Random Forest")
    evaluate_classification(y_test_array, xgb_pred, "XGBoost")

    # Test suggestions with sample inputs
    test_suggestions()

    
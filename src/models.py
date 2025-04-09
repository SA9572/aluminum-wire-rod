# import pandas as pd
# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import cross_val_score, GridSearchCV
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.multioutput import MultiOutputRegressor
# import xgboost as xgb
# import joblib
# import os

# def load_data(data_dir):
#     """Load preprocessed data."""
#     X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
#     X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
#     y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
#     y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))
    
#     print("X_train shape:", X_train.shape)
#     print("X_test shape:", X_test.shape)
#     print("y_train shape:", y_train.shape)
#     print("y_test shape:", y_test.shape)
#     print("X_train sample:\n", X_train.head())
#     print("y_train sample:\n", y_train.head())
    
#     return X_train, X_test, y_train, y_test

# def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
#     """Train and evaluate a model."""
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
#     r2 = r2_score(y_test, y_pred, multioutput='raw_values')
#     cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)

#     print(f"\n{model_name} Results:")
#     print(f"Test MSE: {mse[0]:.4f} (Proof Stress), {mse[1]:.4f} (Tensile Strength)")
#     print(f"Test R²: {r2[0]:.4f} (Proof Stress), {r2[1]:.4f} (Tensile Strength)")
#     print(f"5-Fold CV R²: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")
#     return model

# def tune_model(model, param_grid, X_train, y_train, model_name):
#     """Tune hyperparameters using GridSearchCV."""
#     grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
#     grid_search.fit(X_train, y_train)
#     print(f"\n{model_name} Best Params: {grid_search.best_params_}")
#    # print(f"Best CV R²: {grid_search.best_score_:.4f}")
#     return grid_search.best_estimator_

# if __name__ == "__main__":
#     # Load data
#     data_dir = "C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/data"
#     X_train, X_test, y_train, y_test = load_data(data_dir)

#     # Baseline: Linear Regression (Multi-output)
#     lr_model = MultiOutputRegressor(LinearRegression())
#     lr_trained = evaluate_model(lr_model, X_train, X_test, y_train, y_test, "Linear Regression")

#     # Advanced: Random Forest with optimized parameters
#     rf_model = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1)
#     rf_trained = evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random Forest")

#     # Advanced: XGBoost with optimized parameters
#     xgb_model = xgb.XGBRegressor(random_state=42, n_estimators=200, max_depth=6, learning_rate=0.1, reg_alpha=0.1, reg_lambda=1.0)
#     xgb_trained = evaluate_model(xgb_model, X_train, X_test, y_train, y_test, "XGBoost")

#     # Hyperparameter Tuning for Random Forest
#     rf_param_grid = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [10, 20, 30],
#         'min_samples_split': [2, 5],
#         'min_samples_leaf': [1, 2]
#     }
#     rf_tuned = tune_model(rf_model, rf_param_grid, X_train, y_train, "Random Forest Tuned")

#     # Hyperparameter Tuning for XGBoost
#     xgb_param_grid = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [3, 6, 9],
#         'learning_rate': [0.05, 0.1],
#         'reg_alpha': [0.1, 1.0],
#         'reg_lambda': [1.0, 10.0]
#     }
#     xgb_tuned = tune_model(xgb_model, xgb_param_grid, X_train, y_train, "XGBoost Tuned")

#     # Save tuned models
#     os.makedirs("C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/models", exist_ok=True)
#     joblib.dump(lr_trained, "C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/models/linear_regression.pkl")
#     joblib.dump(rf_tuned, "C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/models/random_forest_tuned.pkl")
#     joblib.dump(xgb_tuned, "C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/models/xgboost_tuned.pkl")
#     print("\nModels saved in C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/models/")

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import joblib
import os

def load_data(data_dir):
    """Load preprocessed data."""
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))
    
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_train shape:", y_train.shape)
    print("y_test shape:", y_test.shape)
    print("X_train sample:\n", X_train.head())
    print("y_train sample:\n", y_train.head())
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Train and evaluate a model."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)

    print(f"\n{model_name} Results:")
    print(f"Test MSE: {mse[0]:.4f} (Proof Stress), {mse[1]:.4f} (Tensile Strength), {mse[2]:.4f} (Elongation)")
    print(f"Test R²: {r2[0]:.4f} (Proof Stress), {r2[1]:.4f} (Tensile Strength), {r2[2]:.4f} (Elongation)")
    print(f"5-Fold CV R²: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")
    return model

def tune_model(model, param_grid, X_train, y_train, model_name):
    """Tune hyperparameters using GridSearchCV."""
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print(f"\n{model_name} Best Params: {grid_search.best_params_}")
    print(f"Best CV R²: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_

if __name__ == "__main__":
    # Load data
    data_dir = "C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/data"
    X_train, X_test, y_train, y_test = load_data(data_dir)

    # Baseline: Linear Regression (Multi-output)
    lr_model = MultiOutputRegressor(LinearRegression())
    lr_trained = evaluate_model(lr_model, X_train, X_test, y_train, y_test, "Linear Regression")

    # Advanced: Random Forest with optimized parameters
    rf_model = RandomForestRegressor(random_state=42, n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1)
    rf_trained = evaluate_model(rf_model, X_train, X_test, y_train, y_test, "Random Forest")

    # Advanced: XGBoost with optimized parameters
    xgb_model = xgb.XGBRegressor(random_state=42, n_estimators=200, max_depth=6, learning_rate=0.1, reg_alpha=0.1, reg_lambda=1.0)
    xgb_trained = evaluate_model(xgb_model, X_train, X_test, y_train, y_test, "XGBoost")

    # Hyperparameter Tuning for Random Forest
    rf_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf_tuned = tune_model(rf_model, rf_param_grid, X_train, y_train, "Random Forest Tuned")

    # Hyperparameter Tuning for XGBoost
    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.05, 0.1],
        'reg_alpha': [0.1, 1.0],
        'reg_lambda': [1.0, 10.0]
    }
    xgb_tuned = tune_model(xgb_model, xgb_param_grid, X_train, y_train, "XGBoost Tuned")

    # Save tuned models
    os.makedirs("C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/models", exist_ok=True)
    joblib.dump(lr_trained, "C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/models/linear_regression.pkl")
    joblib.dump(rf_tuned, "C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/models/random_forest_tuned.pkl")
    joblib.dump(xgb_tuned, "C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/models/xgboost_tuned.pkl")
    print("\nModels saved in C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/models/")


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import xgboost as xgb
import joblib
import os

def load_data(data_dir):
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv'))
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    r2 = r2_score(y_test, y_pred, multioutput='raw_values')
    print(f"\n{model_name} Results:")
    print(f"Test MSE: {mse[0]:.4f} (Proof Stress), {mse[1]:.4f} (Tensile Strength), {mse[2]:.4f} (Elongation)")
    print(f"Test R²: {r2[0]:.4f} (Proof Stress), {r2[1]:.4f} (Tensile Strength), {r2[2]:.4f} (Elongation)")
    return model

if __name__ == "__main__":
    data_dir = "C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/data"
    X_train, X_test, y_train, y_test = load_data(data_dir)

    # XGBoost with extensive hyperparameter tuning
    xgb_model = xgb.XGBRegressor(random_state=42)
    param_grid = {
        'n_estimators': [200, 300, 500],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [1.0, 10.0]
    }
    grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    best_xgb = grid_search.best_estimator_
    
    print(f"Best XGBoost Params: {grid_search.best_params_}")
    evaluate_model(best_xgb, X_train, X_test, y_train, y_test, "Tuned XGBoost")
    
    # Save the tuned model
    os.makedirs("C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/models", exist_ok=True)
    MODEL_PATH = "C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/models/tuned_xgboost.pkl"
    joblib.dump(best_xgb, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
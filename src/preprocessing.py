# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import PolynomialFeatures
# import os

# # Load dataset
# data_path = "C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/data/INT254_dataset_Final.csv"
# df = pd.read_csv('data/INT254_dataset_Final.csv')

# # Define features and targets
# feature_cols = [' C', ' Si', ' Mn', ' P', ' S', ' Ni', ' Cr', ' Mo', ' Cu', 'V', ' Al', ' N', ' Temperature (°C)']
# target_cols = [' 0.2% Proof Stress (MPa)', ' Tensile Strength (MPa)', ' Elongation (%)']

# X = df[feature_cols]
# y = df[target_cols]

# # Feature engineering: Add polynomial features
# poly = PolynomialFeatures(degree=2, include_bias=False)
# X_poly = poly.fit_transform(X)
# X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(feature_cols))

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(X_poly_df, y, test_size=0.2, random_state=42)

# # Standardize features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Convert back to DataFrame with column names preserved
# X_train_df = pd.DataFrame(X_train_scaled, columns=X_poly_df.columns)
# X_test_df = pd.DataFrame(X_test_scaled, columns=X_poly_df.columns)
# y_train_df = pd.DataFrame(y_train, columns=target_cols)  # Ensure column names are kept
# y_test_df = pd.DataFrame(y_test, columns=target_cols)    # Ensure column names are kept

# # Save preprocessed data
# output_dir = "C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/data"
# os.makedirs(output_dir, exist_ok=True)

# X_train_df.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
# X_test_df.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
# y_train_df.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
# y_test_df.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

# print(f"Preprocessing completed! Data saved in {output_dir}")
# print(f"Scaler saved in {output_dir}/scaler.pkl")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
import os
import joblib

# Load dataset
data_path = "C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/data/INT254_dataset_Final.csv"
df = pd.read_csv('data/INT254_dataset_Final.csv')

# Define features and targets
feature_cols = [' C', ' Si', ' Mn', ' P', ' S', ' Ni', ' Cr', ' Mo', ' Cu', 'V', ' Al', ' N', ' Temperature (°C)']
target_cols = [' 0.2% Proof Stress (MPa)', ' Tensile Strength (MPa)', ' Elongation (%)']

# Verify columns exist
missing_features = [col for col in feature_cols if col not in df.columns]
missing_targets = [col for col in target_cols if col not in df.columns]
if missing_features or missing_targets:
    raise KeyError(f"Missing columns in dataset: Features - {missing_features}, Targets - {missing_targets}")

X = df[feature_cols]
y = df[target_cols]

# Feature engineering: Add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(feature_cols))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_poly_df, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame with column names preserved
X_train_df = pd.DataFrame(X_train_scaled, columns=X_poly_df.columns)
X_test_df = pd.DataFrame(X_test_scaled, columns=X_poly_df.columns)
y_train_df = pd.DataFrame(y_train, columns=target_cols)
y_test_df = pd.DataFrame(y_test, columns=target_cols)

# Save preprocessed data
output_dir = "C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/data"
os.makedirs(output_dir, exist_ok=True)

X_train_df.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
X_test_df.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
y_train_df.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
y_test_df.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

# Save scaler for inverse transformation
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
print(f"Preprocessing completed! Data saved in {output_dir}")
print(f"Scaler saved in {output_dir}/scaler.pkl")
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import joblib
import os
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)

# Load the trained model and scaler
MODEL_PATH = "C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/models/xgboost_tuned.pkl"
SCALER_PATH = "C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/data/scaler.pkl"
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Define all feature columns
ALL_FEATURES = ['C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu', 'V', 'Al', 'N', 'Ceq', 'Nb + Ta', 'Temperature (°C)']
INPUT_FEATURES = ['Si', 'Mn', 'Cu', 'Al', 'Temperature (°C)']
poly = PolynomialFeatures(degree=2, include_bias=False)

def classify_quality(tensile_strength, elongation, tensile_threshold=400, elongation_threshold=25):
    if tensile_strength > tensile_threshold and elongation > elongation_threshold:
        return 1  # Better
    return 0  # Not Better

def suggest_improvements(tensile_strength, elongation, tensile_threshold=400, elongation_threshold=25):
    if tensile_strength <= tensile_threshold and elongation <= elongation_threshold:
        return "Increase annealing time and temperature to improve both tensile strength and ductility."
    elif tensile_strength <= tensile_threshold:
        return "Increase annealing time or adjust alloy composition (e.g., add Si or Mn)."
    elif elongation <= elongation_threshold:
        return "Reduce cold working or increase annealing temperature to enhance ductility."
    return "Quality meets standards; no improvements needed."

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            inputs = {
                'Si': float(request.form.get('Si', 0)),
                'Mn': float(request.form.get('Mn', 0)),
                'Cu': float(request.form.get('Cu', 0)),
                'Al': float(request.form.get('Al', 0)),
                'Temperature (°C)': float(request.form.get('Temperature', 0))
            }
            print(f"Parsed inputs: {inputs}")
            
            full_inputs = {
                'C': 0.05, 'Si': inputs['Si'], 'Mn': inputs['Mn'], 'P': 0.015, 'S': 0.01,
                'Ni': 0.03, 'Cr': 0.05, 'Mo': 0.01, 'Cu': inputs['Cu'], 'V': 0.001,
                'Al': inputs['Al'], 'N': 0.015, 'Ceq': 0.4, 'Nb + Ta': 0.02,
                'Temperature (°C)': inputs['Temperature (°C)']
            }
            
            input_df = pd.DataFrame([full_inputs], columns=ALL_FEATURES)
            input_poly = poly.fit_transform(input_df)
            input_poly_df = pd.DataFrame(input_poly, columns=poly.get_feature_names_out(ALL_FEATURES))
            
            input_scaled = scaler.transform(input_poly_df)
            prediction = model.predict(input_scaled)[0]
            proof_stress, tensile_strength, elongation = prediction
            
            # Redirect to quality page with predicted values
            return redirect(url_for('quality', tensile_strength=tensile_strength, elongation=elongation))
        
        except ValueError:
            return render_template('index.html', error="Please enter valid numeric values for all fields.")
        except Exception as e:
            return render_template('index.html', error=f"Error: {str(e)}")
    
    return render_template('index.html')

@app.route('/quality', methods=['GET', 'POST'])
def quality():
    if request.method == 'POST':
        try:
            tensile_strength = float(request.form.get('tensile_strength', 0))
            elongation = float(request.form.get('elongation', 0))
            
            quality = classify_quality(tensile_strength, elongation)
            quality_message = ("This aluminum wire rod quality is better" if quality == 1 else 
                              "This aluminum wire rod quality is not better for industry uses and you need to improve your quality")
            suggestion = suggest_improvements(tensile_strength, elongation)
            
            return render_template('quality.html', quality_message=quality_message, suggestion=suggestion,
                                 tensile_strength=tensile_strength, elongation=elongation)
        
        except ValueError:
            return render_template('quality.html', error="Please enter valid numeric values.")
    
    # Handle GET request (from redirect)
    tensile_strength = float(request.args.get('tensile_strength', 0))
    elongation = float(request.args.get('elongation', 0))
    quality = classify_quality(tensile_strength, elongation)
    quality_message = ("This aluminum wire rod quality is better" if quality == 1 else 
                      "This aluminum wire rod quality is not better for industry uses and you need to improve your quality")
    suggestion = suggest_improvements(tensile_strength, elongation)
    
    return render_template('quality.html', quality_message=quality_message, suggestion=suggestion,
                         tensile_strength=tensile_strength, elongation=elongation)

if __name__ == '__main__':
    app.run(debug=True)
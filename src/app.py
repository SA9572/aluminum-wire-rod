# # import pandas as pd
# # import numpy as np
# # from flask import Flask, render_template, request, redirect, url_for
# # import joblib
# # import os
# # from sklearn.preprocessing import PolynomialFeatures

# # app = Flask(__name__)

# # # Load the trained model and scaler
# # MODEL_PATH = "C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/models/xgboost_tuned.pkl"
# # SCALER_PATH = "C:/Users/ssaur/OneDrive/Desktop/aluminum-wire-rod/data/scaler.pkl"
# # model = joblib.load(MODEL_PATH)
# # scaler = joblib.load(SCALER_PATH)

# # # Define all feature columns
# # ALL_FEATURES = ['C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu', 'V', 'Al', 'N', 'Ceq', 'Nb + Ta', 'Temperature (°C)']
# # INPUT_FEATURES = ['Si', 'Mn', 'Cu', 'Al', 'Temperature (°C)']
# # poly = PolynomialFeatures(degree=2, include_bias=False)

# # def classify_quality(tensile_strength, elongation, tensile_threshold=400, elongation_threshold=25):
# #     if tensile_strength > tensile_threshold and elongation > elongation_threshold:
# #         return 1  # Better
# #     return 0  # Not Better

# # def suggest_improvements(tensile_strength, elongation, tensile_threshold=400, elongation_threshold=25):
# #     if tensile_strength <= tensile_threshold and elongation <= elongation_threshold:
# #         return "Increase annealing time and temperature to improve both tensile strength and ductility."
# #     elif tensile_strength <= tensile_threshold:
# #         return "Increase annealing time or adjust alloy composition (e.g., add Si or Mn)."
# #     elif elongation <= elongation_threshold:
# #         return "Reduce cold working or increase annealing temperature to enhance ductility."
# #     return "Quality meets standards; no improvements needed."

# # @app.route('/', methods=['GET', 'POST'])
# # def index():
# #     if request.method == 'POST':
# #         try:
# #             inputs = {
# #                 'Si': float(request.form.get('Si', 0)),
# #                 'Mn': float(request.form.get('Mn', 0)),
# #                 'Cu': float(request.form.get('Cu', 0)),
# #                 'Al': float(request.form.get('Al', 0)),
# #                 'Temperature (°C)': float(request.form.get('Temperature', 0))
# #             }
# #             print(f"Parsed inputs: {inputs}")
            
# #             full_inputs = {
# #                 'C': 0.05, 'Si': inputs['Si'], 'Mn': inputs['Mn'], 'P': 0.015, 'S': 0.01,
# #                 'Ni': 0.03, 'Cr': 0.05, 'Mo': 0.01, 'Cu': inputs['Cu'], 'V': 0.001,
# #                 'Al': inputs['Al'], 'N': 0.015, 'Ceq': 0.4, 'Nb + Ta': 0.02,
# #                 'Temperature (°C)': inputs['Temperature (°C)']
# #             }
            
# #             input_df = pd.DataFrame([full_inputs], columns=ALL_FEATURES)
# #             input_poly = poly.fit_transform(input_df)
# #             input_poly_df = pd.DataFrame(input_poly, columns=poly.get_feature_names_out(ALL_FEATURES))
            
# #             input_scaled = scaler.transform(input_poly_df)
# #             prediction = model.predict(input_scaled)[0]
# #             proof_stress, tensile_strength, elongation = prediction
            
# #             # Redirect to quality page with predicted values
# #             return redirect(url_for('quality', tensile_strength=tensile_strength, elongation=elongation))
        
# #         except ValueError:
# #             return render_template('inde.html', error="Please enter valid numeric values for all fields.")
# #         except Exception as e:
# #             return render_template('inde.html', error=f"Error: {str(e)}")
    
# #     return render_template('inde.html')

# # @app.route('/quality', methods=['GET', 'POST'])
# # def quality():
# #     if request.method == 'POST':
# #         try:
# #             tensile_strength = float(request.form.get('tensile_strength', 0))
# #             elongation = float(request.form.get('elongation', 0))
            
# #             quality = classify_quality(tensile_strength, elongation)
# #             quality_message = ("This aluminum wire rod quality is better" if quality == 1 else 
# #                               "This aluminum wire rod quality is not better for industry uses and you need to improve your quality")
# #             suggestion = suggest_improvements(tensile_strength, elongation)
            
# #             return render_template('quality.html', quality_message=quality_message, suggestion=suggestion,
# #                                  tensile_strength=tensile_strength, elongation=elongation)
        
# #         except ValueError:
# #             return render_template('quality.html', error="Please enter valid numeric values.")
    
# #     # Handle GET request (from redirect)
# #     tensile_strength = float(request.args.get('tensile_strength', 0))
# #     elongation = float(request.args.get('elongation', 0))
# #     quality = classify_quality(tensile_strength, elongation)
# #     quality_message = ("This aluminum wire rod quality is better" if quality == 1 else 
# #                       "This aluminum wire rod quality is not better for industry uses and you need to improve your quality")
# #     suggestion = suggest_improvements(tensile_strength, elongation)
    
# #     return render_template('quality.html', quality_message=quality_message, suggestion=suggestion,
# #                          tensile_strength=tensile_strength, elongation=elongation)

# # if __name__ == '__main__':
# #     app.run(debug=True)



# # app.py
# import pandas as pd
# import numpy as np
# from flask import Flask, render_template, request, redirect, url_for
# import joblib
# import os
# from sklearn.preprocessing import PolynomialFeatures

# app = Flask(__name__)

# # Load the trained model and scaler
# MODEL_PATH = "models/xgboost_tuned.pkl"
# SCALER_PATH = "data/scaler.pkl"
# model = joblib.load(MODEL_PATH)
# scaler = joblib.load(SCALER_PATH)

# ALL_FEATURES = ['C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu', 'V', 'Al', 'N', 'Ceq', 'Nb + Ta', 'Temperature (°C)']
# inputs = ['Si', 'Mn', 'Cu', 'Al', 'Temperature (°C)']
# poly = PolynomialFeatures(degree=2, include_bias=False)

# def classify_quality(tensile_strength, elongation, tensile_threshold=400, elongation_threshold=25):
#     if tensile_strength > tensile_threshold and elongation > elongation_threshold:
#         return 1
#     return 0

# def suggest_improvements(tensile_strength, elongation):
#     if tensile_strength <= 400 and elongation <= 25:
#         return "Increase annealing time and temperature to improve both tensile strength and ductility."
#     elif tensile_strength <= 400:
#         return "Increase annealing time or adjust alloy composition (e.g., add Si or Mn)."
#     elif elongation <= 25:
#         return "Reduce cold working or increase annealing temperature to enhance ductility."
#     return "Quality meets standards; no improvements needed."

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         try:
#             inputs = {
#                 'Si': float(request.form.get('Si')),
#                 'Mn': float(request.form.get('Mn')),
#                 'Cu': float(request.form.get('Cu')),
#                 'Al': float(request.form.get('Al')),
#                 'Temperature (°C)': float(request.form.get('Temperature'))
#             }
#         except ValueError as ve:
#             return render_template('inde.html', error="Invalid input: Please enter numeric values.")
#         except Exception as e:
#             return render_template('inde.html', error=f"Unexpected error: {str(e)}")

#             # full_inputs = {
#             #     'C': 0.05, 'Si': inputs['Si'], 'Mn': inputs['Mn'], 'P': 0.015, 'S': 0.01,
#             #     'Ni': 0.03, 'Cr': 0.05, 'Mo': 0.01, 'Cu': inputs['Cu'], 'V': 0.001,
#             #     'Al': inputs['Al'], 'N': 0.015, 'Ceq': 0.4, 'Nb + Ta': 0.02,
#             #     'Temperature (°C)': inputs['Temperature (°C)']
#             # }
#             # Ensure the input features match the training features
# full_inputs = {
#     'C': 0.05, 'Si': inputs['Si'], 'Mn': inputs['Mn'], 'P': 0.015, 'S': 0.01,
#     'Ni': 0.03, 'Cr': 0.05, 'Mo': 0.01, 'Cu': inputs['Cu'], 'V': 0.001,
#     'Al': inputs['Al'], 'N': 0.015, 'Ceq': 0.4, 'Nb + Ta': 0.02,
#     'Temperature (°C)': inputs['Temperature (°C)']
# }

# # Create DataFrame with the exact feature names used during training
# input_df = pd.DataFrame([full_inputs], columns=ALL_FEATURES)

# # Apply the same preprocessing steps as during training
# input_poly = poly.fit_transform(input_df)
# input_poly_df = pd.DataFrame(input_poly, columns=poly.get_feature_names_out(ALL_FEATURES))

# # Scale the data using the trained scaler
# input_scaled = scaler.transform(input_poly_df)

# # Predict using the trained model
# prediction = model.predict(input_scaled)

#             input_df = pd.DataFrame([full_inputs], columns=ALL_FEATURES)
#             input_poly = poly.fit_transform(input_df)
#             input_poly_df = pd.DataFrame(input_poly, columns=poly.get_feature_names_out(ALL_FEATURES))
#             input_scaled = scaler.transform(input_poly_df)
#             prediction = model.predict(input_scaled)

#             # Fix: unpack numpy array properly
#             proof_stress, tensile_strength, elongation = prediction.tolist()[0]

#             # Ensure return is inside the function
#             return redirect(url_for('quality', tensile_strength=tensile_strength, elongation=elongation))
        
#         except Exception as e:
#             return render_template('inde.html', error=f"Error: {str(e)}")

#     return render_template('inde.html')

# @app.route('/quality', methods=['GET', 'POST'])
# def quality():
#     tensile_strength = float(request.args.get('tensile_strength', 0))
#     elongation = float(request.args.get('elongation', 0))
#     quality = classify_quality(tensile_strength, elongation)
#     quality_message = "✅ This aluminum wire rod quality is better" if quality == 1 else "❌ This aluminum wire rod quality is not better for industry use"
#     suggestion = suggest_improvements(tensile_strength, elongation)
    
#     return render_template('quality.html',
#                            tensile_strength=round(tensile_strength, 2),
#                            elongation=round(elongation, 2),
#                            quality_message=quality_message,
#                            suggestion=suggestion)

# if __name__ == '__main__':
#     app.run(debug=True)


import pandas as pd
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
import joblib
import os
from sklearn.preprocessing import PolynomialFeatures

app = Flask(__name__)

# Load the trained model and scaler
MODEL_PATH = "models/xgboost_tuned.pkl"
SCALER_PATH = "data/scaler.pkl"
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Define all feature columns
ALL_FEATURES = ['C', 'Si', 'Mn', 'P', 'S', 'Ni', 'Cr', 'Mo', 'Cu', 'V', 'Al', 'N', 'Ceq', 'Nb + Ta', 'Temperature (°C)']
poly = PolynomialFeatures(degree=2, include_bias=False)

def classify_quality(tensile_strength, elongation, tensile_threshold=400, elongation_threshold=25):
    if tensile_strength > tensile_threshold and elongation > elongation_threshold:
        return 1
    return 0

def suggest_improvements(tensile_strength, elongation):
    if tensile_strength <= 400 and elongation <= 25:
        return "Increase annealing time and temperature to improve both tensile strength and ductility."
    elif tensile_strength <= 400:
        return "Increase annealing time or adjust alloy composition (e.g., add Si or Mn)."
    elif elongation <= 25:
        return "Reduce cold working or increase annealing temperature to enhance ductility."
    return "Quality meets standards; no improvements needed."

# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         try:
#             # Parse inputs from the form
#             inputs = {
#                 'Si': float(request.form.get('Si', 0)),
#                 'Mn': float(request.form.get('Mn', 0)),
#                 'Cu': float(request.form.get('Cu', 0)),
#                 'Al': float(request.form.get('Al', 0)),
#                 'Temperature (°C)': float(request.form.get('Temperature', 0))
#             }
#             print(f"Parsed inputs: {inputs}")
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         try:
#             # Log raw form data
#             print("Raw form data:", request.form)

#             # Parse inputs from the form
#             inputs = {
#                 'Si': float(request.form.get('Si', 0)),
#                 'Mn': float(request.form.get('Mn', 0)),
#                 'Cu': float(request.form.get('Cu', 0)),
#                 'Al': float(request.form.get('Al', 0)),
#                 'Temperature (°C)': float(request.form.get('Temperature', 0))
#             }
#             print(f"Parsed inputs: {inputs}")
            
#             # Fill missing features with default values
#             full_inputs = {
#                 'C': 0.05, 'Si': inputs['Si'], 'Mn': inputs['Mn'], 'P': 0.015, 'S': 0.01,
#                 'Ni': 0.03, 'Cr': 0.05, 'Mo': 0.01, 'Cu': inputs['Cu'], 'V': 0.001,
#                 'Al': inputs['Al'], 'N': 0.015, 'Ceq': 0.4, 'Nb + Ta': 0.02,
#                 'Temperature (°C)': inputs['Temperature (°C)']
#             }

#             # Create DataFrame with the exact feature names used during training
#             input_df = pd.DataFrame([full_inputs], columns=ALL_FEATURES)

#             # Apply the same preprocessing steps as during training
#             input_poly = poly.fit_transform(input_df)
#             input_poly_df = pd.DataFrame(input_poly, columns=poly.get_feature_names_out(ALL_FEATURES))

#             # Scale the data using the trained scaler
#             input_scaled = scaler.transform(input_poly_df)

#             # Predict using the trained model
#             prediction = model.predict(input_scaled)

#             # Unpack the prediction
#             proof_stress, tensile_strength, elongation = prediction.tolist()[0]

#             # Redirect to the quality page with predicted values
#             return redirect(url_for('quality', tensile_strength=tensile_strength, elongation=elongation))
        
#         except ValueError:
#             return render_template('inde.html', error="Please enter valid numeric values for all fields.")
#         except Exception as e:
#             return render_template('inde.html', error=f"Error: {str(e)}")
    
#     return render_template('inde.html')
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Log raw form data for debugging
            print("Raw form data:", request.form)

            # Parse inputs from the form
            inputs = {
                'Si': float(request.form.get('Si', 0)),
                'Mn': float(request.form.get('Mn', 0)),
                'Cu': float(request.form.get('Cu', 0)),
                'Al': float(request.form.get('Al', 0)),
                'Temperature (°C)': float(request.form.get('Temperature', 0))
            }

            # Check if any field is missing or invalid
            if not all(inputs.values()):
                raise ValueError("All fields are required and must be numeric.")

            print(f"Parsed inputs: {inputs}")

            # Proceed with prediction logic
            full_inputs = {
                'C': 0.05, 'Si': inputs['Si'], 'Mn': inputs['Mn'], 'P': 0.015, 'S': 0.01,
                'Ni': 0.03, 'Cr': 0.05, 'Mo': 0.01, 'Cu': inputs['Cu'], 'V': 0.001,
                'Al': inputs['Al'], 'N': 0.015, 'Ceq': 0.4, 'Nb + Ta': 0.02,
                'Temperature (°C)': inputs['Temperature (°C)']
            }

            # Create DataFrame with the exact feature names used during training
            input_df = pd.DataFrame([full_inputs], columns=ALL_FEATURES)

            # Apply preprocessing steps
            input_poly = poly.fit_transform(input_df)
            input_poly_df = pd.DataFrame(input_poly, columns=poly.get_feature_names_out(ALL_FEATURES))
            input_scaled = scaler.transform(input_poly_df)

            # Predict using the trained model
            prediction = model.predict(input_scaled)
            proof_stress, tensile_strength, elongation = prediction.tolist()[0]

            # Redirect to quality page with predicted values
            return redirect(url_for('quality', tensile_strength=tensile_strength, elongation=elongation))

        except ValueError as ve:
            print(f"ValueError: {ve}")
            return render_template('inde.html', error="Please enter valid numeric values for all fields.")
        except Exception as e:
            print(f"Exception: {e}")
            return render_template('inde.html', error=f"Unexpected error: {str(e)}")

    return render_template('inde.html')

@app.route('/quality', methods=['GET', 'POST'])
def quality():
    tensile_strength = float(request.args.get('tensile_strength', 0))
    elongation = float(request.args.get('elongation', 0))
    quality = classify_quality(tensile_strength, elongation)
    quality_message = "✅ This aluminum wire rod quality is better" if quality == 1 else "❌ This aluminum wire rod quality is not better for industry use"
    suggestion = suggest_improvements(tensile_strength, elongation)
    
    return render_template('quality.html',
                           tensile_strength=round(tensile_strength, 2),
                           elongation=round(elongation, 2),
                           quality_message=quality_message,
                           suggestion=suggestion)

if __name__ == '__main__':
    app.run(debug=True)


# from flask import Flask, request, render_template, jsonify
# import pickle
# import numpy as np
# import logging

# app = Flask(__name__, static_url_path='/static', static_folder='static', template_folder='templates')

# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)

# try:
#     with open('model.pkl', 'rb') as file:
#         model = pickle.load(file)
#     logger.info("Model loaded successfully")
# except Exception as e:
#     logger.error(f"Failed to load model: {str(e)}")
#     raise

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/about')
# def about():
#     return render_template('about.html')

# @app.route('/properties')
# def properties():
#     return render_template('properties.html')

# @app.route('/types')
# def types():
#     return render_template('types.html')

# @app.route('/applications')
# def applications():
#     return render_template('applications.html')

# @app.route('/faqs')
# def faqs():
#     return render_template('faqs.html')

# @app.route('/resources')
# def resources():
#     return render_template('resources.html')

# @app.route('/blogs')
# def blog():
#     return render_template('blog.html')

# @app.route('/contact')
# def contact():
#     return render_template('contact.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     logger.debug("Received predict request")
#     try:
#         tensile_strength = float(request.form['tensile_strength'])
#         proof_stress = float(request.form['proof_stress'])
#         elongation = float(request.form['elongation'])

#         if tensile_strength <= 0 or proof_stress <= 0 or elongation <= 0:
#             logger.warning("Invalid input: Negative or zero values")
#             return jsonify({'error': 'All inputs must be positive numbers'}), 400

#         input_data = np.array([[tensile_strength, proof_stress, elongation]])
#         logger.debug(f"Input data: {input_data}")

#         prediction = model.predict(input_data)[0]
#         confidence = model.predict_proba(input_data).max() * 100
#         logger.debug(f"Prediction: {prediction}, Confidence: {confidence}")

#         if prediction == 1:
#             result = "Good for Industry Use"
#             suggestion = "No improvements needed."
#         else:
#             result = "Not Suitable for Industry Use"
#             suggestion = "Consider increasing tensile strength or adjusting heat treatment."

#         return jsonify({
#             'result': result,
#             'confidence': f"{confidence:.2f}%",
#             'suggestion': suggestion,
#             'tensile_strength': tensile_strength,
#             'proof_stress': proof_stress,
#             'elongation': elongation
#         })
#     except Exception as e:
#         logger.error(f"Prediction error: {str(e)}")
#         return jsonify({'error': str(e)}), 400

# if __name__ == '__main__':
#     app.run(debug=True)


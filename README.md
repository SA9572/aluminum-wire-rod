# AI-Driven Assessment of Aluminum Wire Rod Properties
A machine learning project to predict tensile strength and elongation of aluminum wire rods.

## Phase 1: Progress
- **Dataset Loaded**: 150+ rows, 19 columns (Alloy code, C, Si, ..., Tensile Strength, Elongation).
- **EDA Completed**: 
  - No missing values.
  - Tensile Strength decreases with temperature.
  - Elongation increases with temperature.
- **Next Steps**: Preprocessing and feature engineering.

# aluminum-wire-rod/
# ├── app.py                 # Flask application
# ├── train_model.py         # Script to train and save the ML model
# ├── model.pkl              # Saved ML model
# ├── predictions.db         # SQLite database
# ├── requirements.txt       # Python dependencies
# ├── templates/             # HTML templates
# │   ├── home.html
# │   ├── aboutus.html
# │   ├── Application.html
# │   ├── product.html
# │   ├── blog.html
# │   ├── Prediction.html
# │   └── logs.html
# ├── static/                # Static files (CSS, JS, images)
# │   └── image/
# └── README.md              # Project documentation
  
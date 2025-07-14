from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model + label encoder
model = joblib.load("injury_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

X_common = [
    "INVTYPE_Passenger", "INVTYPE_Pedestrian", "INVTYPE_Driver",
    "IMPACTYPE_Pedestrian Collisions", "IMPACTYPE_Cyclist Collisions",
    "ACCLASS_Non-Fatal Injury", "INVAGE_85 to 89", "INVAGE_80 to 84"
]

@app.route('/')
def home():
    return render_template('index.html', feature_names=X_common)

@app.route('/predict', methods=['POST'])
def predict():
    input_data = {col: int(request.form.get(col, 0)) for col in X_common}
    input_df = pd.DataFrame([input_data])

    prediction_code = model.predict(input_df)[0]  # returns 0, 1, etc.
    prediction_label = label_encoder.inverse_transform([prediction_code])[0]  # → 'Fatal'

    return render_template("result.html", prediction_text=f"Injury Severity: {prediction_label}")

if __name__ == '__main__':
    app.run(debug=True)

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input features from form
        feature_names = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
                         'ejection_fraction', 'high_blood_pressure', 'platelets',
                         'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']

        features = [float(request.form[name]) for name in feature_names]

        # Scale and predict
        input_scaled = scaler.transform([features])
        prediction = model.predict(input_scaled)[0]
        result = "🔴 Patient is at risk of death." if prediction == 1 else "🟢 Patient is likely to survive."
    
    except Exception as e:
        result = f"⚠️ Error occurred: {e}"
    
    return render_template('index.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)

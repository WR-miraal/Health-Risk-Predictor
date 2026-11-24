
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# -----------------------------------
# Load trained model and encoders
# -----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "health_risk_predictor.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    encoders = pickle.load(f)


# -----------------------------------
# Home Page
# -----------------------------------

@app.route("/")
def home():
    return render_template("index.html")


# -----------------------------------
# Prediction API
# -----------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:
        data = request.get_json()

        # Read numerical inputs
        age = int(data["age"])
        exercise_days = int(data["exercise_days"])
        sleep = int(data["sleep"])
        bmi = float(data["bmi"])

        # Read categorical inputs
        diet = data["diet"]
        stress = data["stress"]
        smoking = data["smoking"]
        alcohol = data["alcohol"]
        family_history = data["family_history"]

        # Validate numerical inputs
        if not 18 <= age <= 80:
            raise ValueError("Age must be between 18 and 80.")

        if not 0 <= exercise_days <= 7:
            raise ValueError("Exercise days must be between 0 and 7.")

        if not 2 <= sleep <= 12:
            raise ValueError("Sleep hours must be between 2 and 12.")

        if not 10 <= bmi <= 40:
            raise ValueError("BMI must be between 10 and 40.")

        # -----------------------------------
        # Encode categorical features
        # -----------------------------------

        diet_encoded = encoders["diet"].transform([diet])[0]

        stress_encoded = encoders["stress"].transform([stress])[0]

        smoking_encoded = encoders["smoking"].transform([smoking])[0]

        alcohol_encoded = encoders["alcohol"].transform([alcohol])[0]

        family_history_encoded = encoders["family_history"].transform(
            [family_history]
        )[0]

        # -----------------------------------
        # Prepare input data
        # Keep the original feature order
        # -----------------------------------

        input_data = [[
            age,
            diet_encoded,
            exercise_days,
            sleep,
            stress_encoded,
            bmi,
            smoking_encoded,
            alcohol_encoded,
            family_history_encoded
        ]]

        input_array = np.array(input_data)

        # -----------------------------------
        # Make Prediction
        # -----------------------------------

        prediction = model.predict(input_array)

        probabilities = model.predict_proba(input_array)

        risk_label = encoders["risk_level"].inverse_transform(
            [prediction[0]]
        )[0]

        # Model confidence percentage
        risk_percentage = float(np.max(probabilities[0]) * 100)

        # -----------------------------------
        # Lifestyle factors for bar chart
        # -----------------------------------

        factors = {
            "Diet": int(diet_encoded),
            "Exercise Days": exercise_days,
            "Sleep Hours": sleep,
            "Stress": int(stress_encoded),
            "BMI": bmi
        }

        return jsonify({
            "success": True,
            "risk_label": str(risk_label),
            "risk_percentage": round(risk_percentage, 2),
            "factors": factors
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


# -----------------------------------
# Run Flask Application
# -----------------------------------

if __name__ == "__main__":
    app.run(debug=True)

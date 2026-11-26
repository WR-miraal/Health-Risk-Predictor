
from pathlib import Path
import pickle
import logging

import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "health_risk_predictor.pkl"
ENCODER_PATH = BASE_DIR / "label_encoder.pkl"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------
# Load Model and Encoders
# -----------------------------------------

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

with open(ENCODER_PATH, "rb") as file:
    encoders = pickle.load(file)


# -----------------------------------------
# Encoder Helper Functions
# -----------------------------------------

def get_encoder(feature_name):

    aliases = {
        "diet": ["diet", "diet_encoder"],
        "stress": ["stress", "stress_level", "stress_encoder"],
        "smoking": ["smoking", "smoking_encoder"],
        "alcohol": ["alcohol", "alcohol_encoder"],
        "family_history": [
            "family_history",
            "familyhistory",
            "family_history_encoder"
        ],
        "risk_level": [
            "risk_level",
            "risk",
            "target",
            "label",
            "risk_level_encoder"
        ]
    }

    if not isinstance(encoders, dict):
        raise TypeError(
            "label_encoder.pkl must contain a dictionary of encoders."
        )

    for key in aliases.get(feature_name, [feature_name]):
        if key in encoders:
            return encoders[key]

    lower_map = {
        str(key).lower(): value
        for key, value in encoders.items()
    }

    for key in aliases.get(feature_name, [feature_name]):
        if key.lower() in lower_map:
            return lower_map[key.lower()]

    raise KeyError(
        f"Encoder not found for {feature_name}. "
        f"Available keys: {list(encoders.keys())}"
    )


def encode_category(feature_name, value):

    encoder = get_encoder(feature_name)

    try:
        return int(encoder.transform([str(value)])[0])

    except ValueError:
        raise ValueError(
            f"Invalid {feature_name}: {value}. "
            f"Allowed values: {list(encoder.classes_)}"
        )


def decode_risk_label(class_value):

    try:
        encoder = get_encoder("risk_level")

        return str(
            encoder.inverse_transform([class_value])[0]
        )

    except KeyError:
        return str(class_value)


# -----------------------------------------
# Home Route
# -----------------------------------------

@app.route("/")
def home():

    return render_template("index.html")


# -----------------------------------------
# Prediction Route
# -----------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json(silent=True)

        if not isinstance(data, dict):
            return jsonify({
                "success": False,
                "error": "Invalid JSON input."
            }), 400

        required_fields = [
            "age",
            "diet",
            "exercise_days",
            "sleep",
            "stress",
            "bmi",
            "smoking",
            "alcohol",
            "family_history"
        ]

        missing = [
            field for field in required_fields
            if field not in data
        ]

        if missing:
            return jsonify({
                "success": False,
                "error": f"Missing fields: {', '.join(missing)}"
            }), 400

        # ---------------------------------
        # Numerical Input Validation
        # ---------------------------------

        age = float(data["age"])
        exercise_days = float(data["exercise_days"])
        sleep = float(data["sleep"])
        bmi = float(data["bmi"])

        if not 18 <= age <= 80:
            raise ValueError("Age must be between 18 and 80.")

        if not 0 <= exercise_days <= 7:
            raise ValueError("Exercise days must be between 0 and 7.")

        if not 2 <= sleep <= 12:
            raise ValueError("Sleep must be between 2 and 12 hours.")

        if not 10 <= bmi <= 40:
            raise ValueError("BMI must be between 10 and 40.")

        # ---------------------------------
        # Encode Categorical Features
        # ---------------------------------

        diet_encoded = encode_category(
            "diet", data["diet"]
        )

        stress_encoded = encode_category(
            "stress", data["stress"]
        )

        smoking_encoded = encode_category(
            "smoking", data["smoking"]
        )

        alcohol_encoded = encode_category(
            "alcohol", data["alcohol"]
        )

        family_history_encoded = encode_category(
            "family_history", data["family_history"]
        )

        # ---------------------------------
        # Prepare Input Array
        # ---------------------------------

        # Feature order must match training.

        input_array = np.array([[
            age,
            diet_encoded,
            exercise_days,
            sleep,
            stress_encoded,
            bmi,
            smoking_encoded,
            alcohol_encoded,
            family_history_encoded
        ]], dtype=float)

        # ---------------------------------
        # Make Prediction
        # ---------------------------------

        prediction = model.predict(input_array)[0]

        if not hasattr(model, "predict_proba"):
            raise RuntimeError(
                "The trained model does not support predict_proba()."
            )

        probabilities = model.predict_proba(input_array)[0]

        # ---------------------------------
        # Dynamic Risk Distribution
        # ---------------------------------

        risk_distribution = []

        for class_value, probability in zip(
            model.classes_,
            probabilities
        ):

            class_label = decode_risk_label(class_value)

            risk_distribution.append({
                "label": class_label,
                "percentage": round(
                    float(probability) * 100, 2
                )
            })

        # ---------------------------------
        # Predicted Class and Confidence
        # ---------------------------------

        predicted_label = decode_risk_label(prediction)

        predicted_probability = max(
            probabilities
        ) * 100

        # ---------------------------------
        # JSON Response
        # ---------------------------------

        return jsonify({

            "success": True,

            "risk_label": predicted_label,

            "risk_percentage": round(
                float(predicted_probability), 2
            ),

            "risk_distribution": risk_distribution

        })

    except (ValueError, TypeError, KeyError) as error:

        return jsonify({
            "success": False,
            "error": str(error)
        }), 400

    except Exception:

        logger.exception("Prediction error")

        return jsonify({
            "success": False,
            "error": "Prediction failed. Check the Flask terminal."
        }), 500


# -----------------------------------------
# Run Flask Application
# -----------------------------------------

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )

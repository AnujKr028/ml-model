from flask import Flask, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend requests

# Load the trained ML model
model = joblib.load("ml-model/model.pkl")  # Make sure this is the CatBoost model

@app.route('/')
def home():
    return "✅ Traffic Route Optimizer API is live!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print("📦 Received data:", data)  # For debugging

        # Ensure input is a 2D list of floats
        features = np.array(data["features"], dtype=float).reshape(1, -1)

        print("🔍 Features for prediction:", features)

        prediction = model.predict(features)

        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        print("❌ Error occurred:", str(e))  # Log error to server console
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run()

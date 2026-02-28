from flask import Flask, request, jsonify
import joblib
import numpy as np
import datetime
import os

app = Flask(__name__)

# ==============================
# Load Model and Scaler
# ==============================

model = joblib.load("ids_model.pkl")
scaler = joblib.load("scaler.pkl")

print("Model and Scaler Loaded Successfully ✅")


# ==============================
# Home Route
# ==============================

@app.route("/")
def home():
    return "🚀 AI Intrusion Detection System is Live!"


# ==============================
# Prediction Route
# ==============================

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "features" not in data:
            return jsonify({"error": "No features provided"}), 400

        features = np.array(data["features"]).reshape(1, -1)

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]

        if prediction == 1:
            result = "⚠ Attack Detected"
            alert_message = "ALERT: Intrusion Detected!"
            print("🚨🚨🚨 ALERT! INTRUSION DETECTED 🚨🚨🚨")
        else:
            result = "Normal Traffic"
            alert_message = "Traffic is Normal"

        # ==============================
        # Logging System
        # ==============================

        log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open("ids_logs.txt", "a") as log_file:
            log_file.write(
                f"{log_time} | Features: {data['features']} | Prediction: {result}\n"
            )

        return jsonify({
            "prediction": result,
            "message": alert_message
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==============================
# Render-Compatible Run
# ==============================

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
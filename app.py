from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import datetime
import os

app = Flask(__name__)

model = joblib.load("ids_model.pkl")
scaler = joblib.load("scaler.pkl")

# Global counter
attack_count = 0

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Intrusion Detection System</title>
    <style>
        body { font-family: Arial; text-align: center; margin-top: 50px; }
        input { width: 80%; padding: 10px; margin: 10px; }
        button { padding: 10px 20px; font-size: 16px; }
        .normal { color: green; font-size: 20px; }
        .attack { color: red; font-size: 20px; }
    </style>
</head>
<body>
    <h1>🚀 AI Intrusion Detection System</h1>
    <p>Total Attacks Detected: <b>{{ attack_count }}</b></p>

    <form method="POST" action="/web_predict">
        <input type="text" name="features" placeholder="Enter 41 comma-separated feature values" required>
        <br>
        <button type="submit">Check Traffic</button>
    </form>

    {% if result %}
        <p class="{{ css_class }}">{{ result }}</p>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    return render_template_string(HTML_PAGE, attack_count=attack_count)

@app.route("/web_predict", methods=["POST"])
def web_predict():
    global attack_count

    try:
        feature_text = request.form["features"]
        feature_list = [float(x.strip()) for x in feature_text.split(",")]

        features = np.array(feature_list).reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]

        if prediction == 1:
            result = "⚠ Attack Detected!"
            css_class = "attack"
            attack_count += 1
            print("🚨 ALERT: Intrusion Detected!")
        else:
            result = "Normal Traffic"
            css_class = "normal"

        # Logging
        log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("ids_logs.txt", "a") as f:
            f.write(f"{log_time} | {feature_list} | {result}\n")

        return render_template_string(
            HTML_PAGE,
            result=result,
            css_class=css_class,
            attack_count=attack_count
        )

    except Exception as e:
        return f"Error: {str(e)}"

@app.route("/predict", methods=["POST"])
def predict():
    global attack_count

    try:
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]

        if prediction == 1:
            attack_count += 1
            return jsonify({
                "prediction": "⚠ Attack Detected",
                "message": "ALERT: Intrusion Detected!"
            })
        else:
            return jsonify({
                "prediction": "Normal Traffic",
                "message": "Traffic is Normal"
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

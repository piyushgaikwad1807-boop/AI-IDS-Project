from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import datetime
import os

app = Flask(__name__)

# Load model and scaler
model = joblib.load("ids_model.pkl")
scaler = joblib.load("scaler.pkl")

attack_count = 0
normal_count = 0

# -------------------------------
# SIMPLE PROFESSIONAL DASHBOARD
# -------------------------------

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Intrusion Detection System</title>
    <style>
        body {
            font-family: Arial;
            background-color: #f4f6f9;
            text-align: center;
        }
        h1 {
            margin-top: 30px;
        }
        form {
            margin-top: 20px;
        }
        textarea {
            width: 60%;
            height: 80px;
            padding: 10px;
        }
        button {
            padding: 10px 20px;
            font-size: 16px;
            margin-top: 10px;
        }
        .attack {
            color: red;
            font-weight: bold;
            font-size: 20px;
        }
        .normal {
            color: green;
            font-weight: bold;
            font-size: 20px;
        }
        .stats {
            margin-top: 20px;
            font-size: 18px;
        }
    </style>
</head>
<body>

<h1>🚀 AI Intrusion Detection System</h1>

<div class="stats">
    <p>Total Attacks Detected: <b>{{ attack_count }}</b></p>
    <p>Total Normal Traffic: <b>{{ normal_count }}</b></p>
</div>

<form method="POST" action="/web_predict">
    <p>Enter comma separated feature values (any length):</p>
    <textarea name="features"></textarea><br>
    <button type="submit">Analyze Traffic</button>
</form>

{% if result %}
    <p class="{{ css_class }}">{{ result }}</p>
{% endif %}

</body>
</html>
"""

# -------------------------------
# HOME ROUTE (IMPORTANT)
# -------------------------------

@app.route("/")
def home():
    return render_template_string(
        HTML_PAGE,
        attack_count=attack_count,
        normal_count=normal_count
    )

# -------------------------------
# WEB FORM PREDICTION
# -------------------------------

@app.route("/web_predict", methods=["POST"])
def web_predict():
    global attack_count, normal_count

    try:
        feature_text = request.form["features"]

        # Convert input to float list
        features = [float(x.strip()) for x in feature_text.split(",") if x.strip() != ""]

        # Auto-adjust to 41 features
        if len(features) < 41:
            features += [0] * (41 - len(features))
        elif len(features) > 41:
            features = features[:41]

        features_array = np.array(features).reshape(1, -1)
        scaled = scaler.transform(features_array)
        prediction = model.predict(scaled)[0]

        if prediction == 1:
            attack_count += 1
            result = "⚠ ALERT: Intrusion Detected!"
            css_class = "attack"
        else:
            normal_count += 1
            result = "Normal Traffic"
            css_class = "normal"

        # Logging
        log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("ids_logs.txt", "a") as f:
            f.write(f"{log_time} | {features} | {result}\n")

        return render_template_string(
            HTML_PAGE,
            result=result,
            css_class=css_class,
            attack_count=attack_count,
            normal_count=normal_count
        )

    except Exception as e:
        return f"Error: {str(e)}"

# -------------------------------
# API ENDPOINT (POSTMAN / THUNDER CLIENT)
# -------------------------------

@app.route("/predict", methods=["POST"])
def predict():
    global attack_count, normal_count

    try:
        data = request.get_json()
        features = data["features"]

        # Auto-adjust
        if len(features) < 41:
            features += [0] * (41 - len(features))
        elif len(features) > 41:
            features = features[:41]

        features_array = np.array(features).reshape(1, -1)
        scaled = scaler.transform(features_array)
        prediction = model.predict(scaled)[0]

        if prediction == 1:
            attack_count += 1
            return jsonify({
                "prediction": "Attack Detected",
                "message": "⚠ Intrusion Alert"
            })
        else:
            normal_count += 1
            return jsonify({
                "prediction": "Normal Traffic",
                "message": "Traffic is Safe"
            })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -------------------------------
# RENDER COMPATIBLE RUN
# -------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

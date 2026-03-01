from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np
import datetime
import os

app = Flask(__name__)

model = joblib.load("ids_model.pkl")
scaler = joblib.load("scaler.pkl")

attack_count = 0
normal_count = 0

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Intrusion Detection Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body { font-family: Arial; text-align: center; background-color: #f4f6f9; }
        h1 { margin-top: 30px; }
        .container { width: 80%; margin: auto; }
        input { width: 60px; padding: 5px; margin: 3px; }
        button { padding: 10px 20px; margin-top: 15px; font-size: 16px; }
        .attack { color: red; font-size: 20px; }
        .normal { color: green; font-size: 20px; }
        .stats { margin-top: 20px; font-size: 18px; }
        .grid { display: grid; grid-template-columns: repeat(8, 1fr); gap: 5px; justify-items: center; }
    </style>
</head>
<body>

<h1>🚀 AI Intrusion Detection Dashboard</h1>

<div class="stats">
    <p>Total Attacks: <b>{{ attack_count }}</b></p>
    <p>Total Normal Traffic: <b>{{ normal_count }}</b></p>
</div>

<div class="container">
<form method="POST" action="/web_predict">
    <div class="grid">
        {% for i in range(41) %}
            <input type="number" step="any" name="f{{i}}" placeholder="F{{i}}">
        {% endfor %}
    </div>
    <button type="submit">Analyze Traffic</button>
</form>

{% if result %}
    <p class="{{ css_class }}">{{ result }}</p>
{% endif %}
</div>

<canvas id="trafficChart" width="400" height="200"></canvas>

<script>
var ctx = document.getElementById('trafficChart').getContext('2d');
var trafficChart = new Chart(ctx, {
    type: 'pie',
    data: {
        labels: ['Normal Traffic', 'Attacks'],
        datasets: [{
            data: [{{ normal_count }}, {{ attack_count }}],
            backgroundColor: ['green', 'red']
        }]
    }
});
</script>

</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(
        HTML_PAGE,
        attack_count=attack_count,
        normal_count=normal_count
    )

@app.route("/web_predict", methods=["POST"])
def web_predict():
    global attack_count, normal_count

    try:
        features = []

        for i in range(41):
            value = request.form.get(f"f{i}")
            if value == "" or value is None:
                features.append(0)
            else:
                features.append(float(value))

        # Auto-adjust safeguard
        if len(features) < 41:
            features += [0] * (41 - len(features))
        elif len(features) > 41:
            features = features[:41]

        features_array = np.array(features).reshape(1, -1)
        scaled = scaler.transform(features_array)
        prediction = model.predict(scaled)[0]

        if prediction == 1:
            attack_count += 1
            result = "⚠ Attack Detected!"
            css_class = "attack"
            print("🚨 ALERT: Intrusion Detected!")
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

@app.route("/predict", methods=["POST"])
def predict():
    global attack_count, normal_count

    try:
        data = request.get_json()
        features = data["features"]

        if len(features) < 41:
            features += [0] * (41 - len(features))
        elif len(features) > 41:
            features = features[:41]

        features_array = np.array(features).reshape(1, -1)
        scaled = scaler.transform(features_array)
        prediction = model.predict(scaled)[0]

        if prediction == 1:
            attack_count += 1
            return jsonify({"prediction": "⚠ Attack Detected"})
        else:
            normal_count += 1
            return jsonify({"prediction": "Normal Traffic"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

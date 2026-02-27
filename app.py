from flask import Flask, render_template, request, redirect
import joblib
import numpy as np

app = Flask(__name__)

# Load the ML Model and Scaler
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

safe_count = 0
fraud_count = 0
risk_history = []

@app.route("/", methods=["GET", "POST"])
def home():
    global safe_count, fraud_count, risk_history
    result = None

    if request.method == "POST":
        if "reset" in request.form:
            safe_count, fraud_count, risk_history = 0, 0, []
            return redirect("/")

        # Get data from form
        amount = float(request.form["amount"])
        time = int(request.form["time"])

        # Prepare data for ML (Scaling is crucial!)
        features = np.array([[amount, time]])
        features_scaled = scaler.transform(features)

        # ML Prediction
        # predict_proba returns [prob_safe, prob_fraud]
        risk_probability = model.predict_proba(features_scaled)[0][1]
        risk_score = int(risk_probability * 100)

        # Logic for UI
        status = "Fraud 🚨" if risk_score > 50 else "Safe ✅"
        
        # Explainability (Simple version)
        reasons = []
        if amount > 2500: reasons.append("Transaction exceeds typical volume thresholds.")
        if time < 6 or time > 22: reasons.append("Activity flagged during high-risk hours.")

        if "Fraud" in status:
            fraud_count += 1
        else:
            safe_count += 1

        risk_history.append(risk_score)
        result = {"risk": risk_score, "status": status, "reasons": reasons}

    return render_template("index.html", result=result, safe_count=safe_count, 
                           fraud_count=fraud_count, total=safe_count + fraud_count, 
                           history=risk_history)

if __name__ == "__main__":
    app.run(debug=True)
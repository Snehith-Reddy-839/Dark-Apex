from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/analysis", methods=["GET", "POST"])
def analysis():

    prediction = None
    risk_percent = 0
    risk_class = None
    severity = None
    message = None

    if request.method == "POST":

        gender = float(request.form["gender"])
        hemoglobin = float(request.form["hemoglobin"])
        mch = float(request.form["mch"])
        mchc = float(request.form["mchc"])
        mcv = float(request.form["mcv"])

        input_data = np.array([[gender, hemoglobin, mch, mchc, mcv]])
        input_scaled = scaler.transform(input_data)

        probs = model.predict_proba(input_scaled)[0]
        anemia_probability = probs[1]
        risk_percent = round(anemia_probability * 100, 2)

        # Severity Levels
        if risk_percent < 20:
            severity = "Normal"
            risk_class = "normal"
        elif 20 <= risk_percent < 40:
            severity = "Mid Risk"
            risk_class = "mid"
        elif 40 <= risk_percent < 70:
            severity = "Moderate Risk"
            risk_class = "moderate"
        else:
            severity = "Severe Risk"
            risk_class = "severe"

        if anemia_probability >= 0.5:
            prediction = "Anemia Detected"
            message = "Blood parameters indicate anemia. Medical consultation recommended."
        else:
            prediction = "No Anemia Detected"
            message = "Blood parameters appear within normal range."

    return render_template("analysis.html",
                           prediction=prediction,
                           risk_percent=risk_percent,
                           risk_class=risk_class,
                           severity=severity,
                           message=message)

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__,
            template_folder="../frontend/templates",
            static_folder="../frontend/static")

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    wind_speed = float(request.form["wind_speed"])
    temperature = float(request.form["temperature"])
    humidity = float(request.form["humidity"])
    pressure = float(request.form["pressure"])

    input_data = np.array([[wind_speed, temperature, humidity, pressure]])
    input_data = scaler.transform(input_data)

    prediction = model.predict(input_data)[0]

    return render_template(
        "index.html",
        prediction_text=f"Predicted Energy Output: {prediction:.2f} kW"
    )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)


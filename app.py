import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS


# -----------------------------
# LOAD DATA
# -----------------------------

data = pd.read_csv("tire_test_data_large.csv")

le_road = LabelEncoder()
le_climate = LabelEncoder()
le_steer = LabelEncoder()
le_drive = LabelEncoder()
le_trailer = LabelEncoder()

data["roadtype"] = le_road.fit_transform(data["roadtype"])
data["climate"] = le_climate.fit_transform(data["climate"])

data["steertire"] = le_steer.fit_transform(data["steertire"])
data["drivetire"] = le_drive.fit_transform(data["drivetire"])
data["trailertire"] = le_trailer.fit_transform(data["trailertire"])

X = data[["roadtype", "loadkg", "axles", "climate"]]

steer_model = RandomForestClassifier().fit(X, data["steertire"])
drive_model = RandomForestClassifier().fit(X, data["drivetire"])
trailer_model = RandomForestClassifier().fit(X, data["trailertire"])


# -----------------------------
# PRESSURE MODEL
# -----------------------------

pressure_data = pd.read_csv("fleet_tire_dataset_real.csv")

le_wear = LabelEncoder()

pressure_data["roadtype"] = le_road.fit_transform(pressure_data["roadtype"])
pressure_data["climate"] = le_climate.fit_transform(pressure_data["climate"])
pressure_data["wear_level"] = le_wear.fit_transform(pressure_data["wear_level"])

pressure_data["load_per_axle"] = pressure_data["loadkg"] / pressure_data["axles"]

features = [
    "roadtype",
    "loadkg",
    "axles",
    "climate",
    "temperature",
    "avg_speed",
    "tire_age",
    "wear_level",
    "load_per_axle"
]

X_pressure = pressure_data[features]
y_pressure = pressure_data["optimal_psi"]

pressure_model = RandomForestRegressor()
pressure_model.fit(X_pressure, y_pressure)


# -----------------------------
# FLASK APP
# -----------------------------

app = Flask(__name__)
CORS(app)


# Serve frontend
@app.route("/")
def home():
    return send_from_directory(".", "index.html")


# Serve CSS
@app.route("/styles.css")
def styles():
    return send_from_directory(".", "styles.css")


# Serve JS
@app.route("/script.js")
def script():
    return send_from_directory(".", "script.js")


# Serve images/files automatically
@app.route('/<path:path>')
def static_files(path):
    return send_from_directory('.', path)


# -----------------------------
# PREDICTION API
# -----------------------------

@app.route("/predict", methods=["POST"])
def predict():

    req = request.get_json()

    roadtype = req["roadType"]
    loadkg = int(req["loadKg"])
    axles = int(req["axles"])
    climate = req["climate"]

    temperature = float(req["temperature"])
    speed = float(req["speed"])
    tire_age = float(req["tireAge"])
    wear = req["wearLevel"]

    road_enc = le_road.transform([roadtype])[0]
    climate_enc = le_climate.transform([climate])[0]
    wear_enc = le_wear.transform([wear])[0]

    X_input = np.array([[road_enc, loadkg, axles, climate_enc]])

    steer = le_steer.inverse_transform(steer_model.predict(X_input))[0]
    drive = le_drive.inverse_transform(drive_model.predict(X_input))[0]
    trailer = le_trailer.inverse_transform(trailer_model.predict(X_input))[0]

    load_per_axle = loadkg / axles

    pressure_input = np.array([[
        road_enc,
        loadkg,
        axles,
        climate_enc,
        temperature,
        speed,
        tire_age,
        wear_enc,
        load_per_axle
    ]])

    pressure = pressure_model.predict(pressure_input)[0]

    if wear == "high" or tire_age > 3:
        failureRisk = "HIGH"

    elif pressure > 115:
        failureRisk = "MEDIUM"

    else:
        failureRisk = "LOW"

    if failureRisk == "HIGH":
        advice = "Replace tire immediately"

    elif failureRisk == "MEDIUM":
        advice = "Check tire pressure soon"

    else:
        advice = "Tire condition safe"

    return jsonify({
        "steerTire": steer,
        "driveTire": drive,
        "trailerTire": trailer,
        "pressure": round(float(pressure), 2),
        "failureRisk": failureRisk,
        "loadPerAxle": round(load_per_axle, 2),
        "safetyAdvice": advice
    })


if __name__ == "__main__":
    app.run(debug=True)

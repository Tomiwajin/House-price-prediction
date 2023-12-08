import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler

flask_app = Flask(__name__)
XGBModel = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))
ocean_proximity_mapping = {'<1H OCEAN': 0, 'INLAND': 1, 'NEAR OCEAN': 2, 'NEAR BAY': 3, 'ISLAND': 4}

@flask_app.route("/")
def index():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    features = [float(request.form[x]) for x in ['longitude', 'latitude', 'House_Age', 'TotalRoomsInBlock', 'TotalBedroomsInBlock', 'PopulationInBlock', 'households', 'MedianIncomeInBlock']]
    ocean_proximity = request.form['ocean_proximity']
    ocean_proximity_numeric = ocean_proximity_mapping.get(ocean_proximity, -1)
    features.append(ocean_proximity_numeric)
    features_scaled = scaler.transform([features])
    result = int(XGBModel.predict(features_scaled)[0])
    return render_template("index.html", pred_result=result)

if __name__ == "__main__":
    flask_app.run(debug=True)

from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("churn_model.pkl")

@app.route("/")
def home():
    return "Churn Prediction API Running!"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    
    features = np.array(data["features"]).reshape(1, -1)
    
    prediction = model.predict(features)
    
    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)
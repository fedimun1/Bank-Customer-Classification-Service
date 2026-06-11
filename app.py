from fastapi import FastAPI
import joblib
import numpy as np


app = FastAPI()

# Load model
model = joblib.load("bank_model.pkl")

# Root test endpoint
@app.get("/")
def home():
    return {"message": "Bank ML Model API is running"}

# Prediction endpoint
@app.post("/predict")
def predict(data: dict):
    try:
        # expected input:
        # total_amount, avg_amount, txn_count, total_credit, total_debit, last_balance

        features = np.array([[
            data["total_amount"],
            data["avg_amount"],
            data["txn_count"],
            data["total_credit"],
            data["total_debit"],
            data["last_balance"]
        ]])

        prediction = model.predict(features)[0]

        return {
            "prediction": int(prediction),
            "result": "High Spender" if prediction == 1 else "Normal Customer"
        }

    except Exception as e:
        return {"error": str(e)}
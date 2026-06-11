from fastapi import FastAPI
import joblib
import numpy as np
import tensorflow as tf

app = FastAPI()

xgb_model = joblib.load("bank_model.pkl")
nn_model = tf.keras.models.load_model("nn_model.h5")
scaler = joblib.load("scaler.pkl")

@app.post("/predict/xgb")
def predict_xgb(data: dict):
    features = np.array([[ 
        data["total_amount"],
        data["avg_amount"],
        data["txn_count"],
        data["total_credit"],
        data["total_debit"],
        data["last_balance"]
    ]])

    pred = xgb_model.predict(features)[0]

    return {
        "model": "XGBoost",
        "prediction": int(pred)
    }


@app.post("/predict/nn")
def predict_nn(data: dict):
    features = np.array([[
        data["total_amount"],
        data["avg_amount"],
        data["txn_count"],
        data["total_credit"],
        data["total_debit"],
        data["last_balance"]
    ]])

    features = scaler.transform(features)
    pred = nn_model.predict(features)[0][0]
    lable = 1 if pred > 0.5 else 0
    result_text = "High Spending Customer" if lable == 1 else "Low Spending Customer"

    return {
        "model": "Neural Network",
        "prediction": lable,
            "result": result_text
    }
  
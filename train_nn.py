import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib

# Load data
data = pd.read_csv("transactions.csv", low_memory=False)

data["CREDIT"] = pd.to_numeric(data["CREDIT"], errors="coerce").fillna(0)
data["DEBIT"] = pd.to_numeric(data["DEBIT"], errors="coerce").fillna(0)

data["amount"] = data["CREDIT"] + data["DEBIT"]

agg = data.groupby("AC_NO").agg({
    "amount": ["sum", "mean", "count"],
    "CREDIT": "sum",
    "DEBIT": "sum",
    "RUN_BAL": "last"
})

agg.columns = [
    "total_amount",
    "avg_amount",
    "txn_count",
    "total_credit",
    "total_debit",
    "last_balance"
]

agg = agg.reset_index()

# Label (same as XGBoost)
agg["label"] = (agg["total_debit"] > agg["total_debit"].median()).astype(int)

X = agg.drop(["AC_NO", "label"], axis=1)
y = agg["label"]

# Scale data (IMPORTANT for neural networks)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

joblib.dump(scaler, "scaler.pkl")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Neural Network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, epochs=20, batch_size=16, validation_split=0.2)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print("Neural Network Accuracy:", accuracy)

# Save model
model.save("nn_model.h5")
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

# 1. Load CSV
data = pd.read_csv("transactions.csv", low_memory=False)

# 2. Clean numeric columns
data["CREDIT"] = pd.to_numeric(data["CREDIT"], errors="coerce").fillna(0)
data["DEBIT"] = pd.to_numeric(data["DEBIT"], errors="coerce").fillna(0)

# 3. Create transaction amount
data["amount"] = data["CREDIT"] + data["DEBIT"]

# 4. Aggregate per customer (IMPORTANT)
agg = data.groupby("AC_NO").agg({
    "amount": ["sum", "mean", "count"],
    "CREDIT": "sum",
    "DEBIT": "sum",
    "RUN_BAL": "last"
})

# flatten columns
agg.columns = [
    "total_amount",
    "avg_amount",
    "txn_count",
    "total_credit",
    "total_debit",
    "last_balance"
]

agg = agg.reset_index()

# 5. Create LABEL (behavior rule)
# high spender = 1, normal = 0
agg["label"] = (agg["total_debit"] > agg["total_debit"].median()).astype(int)

# 6. Features and target
X = agg.drop(["AC_NO", "label"], axis=1)
y = agg["label"]

# 7. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 8. Train model
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1
)

model.fit(X_train, y_train)

# 9. TEST MODEL (IMPORTANT PART)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 10. Save model
joblib.dump(model, "bank_model.pkl")

print("Model saved successfully")
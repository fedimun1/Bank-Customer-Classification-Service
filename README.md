<!-- using python 3 -->

# Bank ML
This repository contains a simple machine learning pipeline and FastAPI service for classifying bank customers into `High Spender` or `Normal Customer` categories.
## Project Structure

- `transactions.csv` - raw transaction data used to build the model
- `train.py` - training script that reads CSV data, aggregates customer transactions, trains an XGBoost classifier, and saves `bank_model.pkl`
- `app.py` - FastAPI application that loads the saved model and exposes a prediction endpoint
## Requirements
- Python 3.10+ (or compatible)
- `venv` or another virtual environment tool

## Installation

1. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install fastapi uvicorn joblib numpy pandas xgboost scikit-learn
```

## Training the Model

Run the training script to build the model and save it as `bank_model.pkl`:

```bash
python train.py
```

This script:

- loads `transactions.csv`
- converts `CREDIT` and `DEBIT` values to numeric
- creates an `amount` column
- aggregates transaction statistics per account
- creates a binary label based on `total_debit`
- trains an `XGBClassifier`
- saves the trained model to `bank_model.pkl`

## Running the API

Start the FastAPI server using Uvicorn:

```bash
uvicorn app:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

- `GET /` - health check endpoint
- `POST /predict` - prediction endpoint

### Prediction Request Example

```json
{
  "total_amount": 12345.67,
  "avg_amount": 234.56,
  "txn_count": 52,
  "total_credit": 8000.0,
  "total_debit": 4345.67,
  "last_balance": 1200.0
}
```

### Prediction Response Example

```json
{
  "prediction": 1,
  "result": "High Spender"
}
```
## Notes

- The model depends on `bank_model.pkl` being present in the repository root.
- If `bank_model.pkl` is missing, run `python train.py` first.
- The classification label is derived from whether `total_debit` is above the median value in the dataset.

inside the project

source venv/bin/activate
pip install fastapi uvicorn joblib numpy pandas scikit-learn xgboost tensorflow
python train.py

# created bank_model.pkl

python train_nn.py

# nn_model.h5

# scaler.pkl

uvicorn app:app --reload

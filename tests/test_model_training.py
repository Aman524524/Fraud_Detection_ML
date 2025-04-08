import os
import pandas as pd
from src.model_training import train_model

def test_train_model(tmp_path):
    # Create fake data
    df = pd.DataFrame({
        "TransactionID": ["T1", "T2", "T3", "T4"],
        "AccountID": ["A1", "A2", "A3", "A4"],
        "TransactionAmount": [100, 150, 200, 250],
        "TransactionDate": ["2023-01-01"] * 4,
        "TransactionType": ["Purchase"] * 4,
        "Location": ["NY"] * 4,
        "DeviceID": ["D1"] * 4,
        "IP Address": ["192.168.1.1"] * 4,
        "MerchantID": ["M1"] * 4,
        "Channel": ["Online"] * 4,
        "CustomerAge": [25, 30, 35, 40],
        "CustomerOccupation": ["Engineer"] * 4,
        "TransactionDuration": [10, 20, 30, 40],
        "LoginAttempts": [1, 2, 1, 3],
        "AccountBalance": [1000, 1100, 900, 1200],
        "PreviousTransactionDate": ["2022-12-31"] * 4
    })

    input_csv = tmp_path / "input.csv"
    output_csv = tmp_path / "output.csv"
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)

    df.to_csv(input_csv, index=False)

    train_model(str(input_csv), str(output_csv), str(model_dir))

    # Assertions
    assert os.path.exists(output_csv)
    assert os.path.exists(model_dir / "isolation_forest_model.pkl")
    assert os.path.exists(model_dir / "scaler.pkl")
    assert os.path.exists(model_dir / "label_encoders.pkl")

    # Check if anomalies were added
    output_df = pd.read_csv(output_csv)
    assert "anomaly" in output_df.columns
    assert output_df["anomaly"].isin([0, 1]).all()

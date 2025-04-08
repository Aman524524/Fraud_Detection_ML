import pandas as pd
from src.data_preprocessing import preprocess_pipeline

def test_preprocess_pipeline():
    # Sample raw data
    data = {
        "TransactionID": ["T1", "T2"],
        "AccountID": ["A1", "A2"],
        "TransactionAmount": [100.0, 250.0],
        "TransactionDate": ["2023-01-01", "2023-01-02"],
        "TransactionType": ["Purchase", "Withdrawal"],
        "Location": ["NY", "LA"],
        "DeviceID": ["D1", "D2"],
        "IP Address": ["192.168.1.1", "10.0.0.1"],
        "MerchantID": ["M1", "M2"],
        "Channel": ["Online", "POS"],
        "CustomerAge": [25, 40],
        "CustomerOccupation": ["Engineer", "Teacher"],
        "TransactionDuration": [30, 45],
        "LoginAttempts": [1, 3],
        "AccountBalance": [1000.0, 500.0],
        "PreviousTransactionDate": ["2022-12-30", "2022-12-31"]
    }
    df = pd.DataFrame(data)

    # Run pipeline
    processed_df = preprocess_pipeline(df.copy())

    # Tests
    assert not processed_df.empty
    assert "TransactionAmount" in processed_df.columns
    assert processed_df.shape[0] == 2

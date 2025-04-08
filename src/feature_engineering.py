import pandas as pd

import pandas as pd

def create_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    # Bin TransactionAmount into 4 quantiles
    if 'TransactionAmount' in df.columns and df['TransactionAmount'].notnull().sum() > 0:
        try:
            df['TransactionAmountBin'] = pd.qcut(df['TransactionAmount'], q=4, labels=False, duplicates='drop')
        except ValueError as e:
            print(f"qcut failed: {e}")
            df['TransactionAmountBin'] = 0  # fallback
    
    # Convert TransactionDate to datetime and extract features
    if 'TransactionDate' in df.columns:
        df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
        df['TransactionHour'] = df['TransactionDate'].dt.hour
        df['TransactionDay'] = df['TransactionDate'].dt.day
        df['TransactionWeekday'] = df['TransactionDate'].dt.weekday
    
    # Convert PreviousTransactionDate and calculate time delta
    if 'PreviousTransactionDate' in df.columns:
        df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'], errors='coerce')
        if 'TransactionDate' in df.columns:
            df['TimeSinceLastTransaction'] = (
                df['TransactionDate'] - df['PreviousTransactionDate']
            ).dt.total_seconds() / 3600.0  # hours
        else:
            df['TimeSinceLastTransaction'] = 0
    
    # Fill any remaining NaNs from datetime operations
    df['TimeSinceLastTransaction'] = df['TimeSinceLastTransaction'].fillna(0)
    df['TransactionHour'] = df['TransactionHour'].fillna(0).astype(int)
    df['TransactionDay'] = df['TransactionDay'].fillna(0).astype(int)
    df['TransactionWeekday'] = df['TransactionWeekday'].fillna(0).astype(int)
    df['TransactionAmountBin'] = df['TransactionAmountBin'].fillna(0).astype(int)

    return df

def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = ['TransactionID', 'AccountID', 'TransactionDate', 'PreviousTransactionDate']
    for col in columns_to_drop:
        if col in df.columns:
            df.drop(columns=col, inplace=True)
    return df

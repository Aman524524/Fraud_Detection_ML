import pandas as pd
import numpy as np

def create_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df['amount_bin'] = pd.qcut(df['amount'], q=4, labels=False, duplicates='drop')

    if 'trans_date_trans_time' in df.columns:
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        df['hour'] = df['trans_date_trans_time'].dt.hour
        df['day'] = df['trans_date_trans_time'].dt.day
        df['weekday'] = df['trans_date_trans_time'].dt.weekday

    if 'customer_id' in df.columns:
        freq = df.groupby('customer_id')['amount'].transform('count')
        df['transaction_count'] = freq

        avg_amount = df.groupby('customer_id')['amount'].transform('mean')
        df['avg_transaction_amount'] = avg_amount

    if 'avg_transaction_amount' in df.columns:
        df['amount_vs_avg'] = df['amount'] - df['avg_transaction_amount']

    df.fillna(0, inplace=True)

    return df

def drop_unnecessary_columns(df: pd.DataFrame, drop_columns: list = None) -> pd.DataFrame:
    if drop_columns is None:
        drop_columns = ['cc_num', 'first', 'last', 'trans_num', 'merchant', 'street', 'city', 'job']

    existing_cols = [col for col in drop_columns if col in df.columns]
    df = df.drop(columns=existing_cols)
    return df

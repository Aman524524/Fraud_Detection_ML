import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna()
    return df

def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def scale_features(df: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(df)

def preprocess_pipeline(file_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    df = load_data(file_path)
    df_cleaned = clean_data(df)
    df_encoded = encode_categorical(df_cleaned)
    scaled_features = scale_features(df_encoded)
    return df_encoded, scaled_features

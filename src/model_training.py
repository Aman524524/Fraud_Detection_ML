import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.data_preprocessing import preprocess_pipeline
from src.feature_engineering import create_transaction_features, drop_unnecessary_columns


def train_model(file_path: str):
    # Step 1: Load and preprocess data
    df, scaled_data = preprocess_pipeline(file_path)

    # Step 2: Feature Engineering
    df = create_transaction_features(df)
    df = drop_unnecessary_columns(df)

    # Step 3: Handle missing values
    df.fillna(0, inplace=True)

    # Step 4: Encode categorical features
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Step 5: Feature Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Step 6: Train Isolation Forest Model
    model = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)
    model.fit(scaled_data)

    # Step 7: Predict anomalies
    df['anomaly'] = model.predict(scaled_data)
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})  # 1 = fraud, 0 = normal

    # Step 8: Save results
    df.to_csv('data/processed/transactions_with_anomaly_labels.csv', index=False)
    print("Labeled data saved to data/processed/transactions_with_anomaly_labels.csv")

    # Step 9: Save model to file
    os.makedirs('models', exist_ok=True)
    model_path = 'models/isolation_forest_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    return model, df

import streamlit as st
import pandas as pd
import joblib
from src.data_preprocessing import preprocess_pipeline
from src.feature_engineering import create_transaction_features, drop_unnecessary_columns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load model
model = joblib.load("models/isolation_forest_model.pkl")

st.title("🔍 Fraud Detection in Financial Transactions")

uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Preprocess + Feature Engineering
    df = create_transaction_features(df)
    df = drop_unnecessary_columns(df)
    df.fillna(0, inplace=True)

    # Encode categorical variables
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    # Predict
    df['anomaly'] = model.predict(scaled_data)
    df['anomaly'] = df['anomaly'].map({1: 0, -1: 1})

    # Display top few transactions
    st.write("### 🧾 Transactions with Anomaly Prediction")
    st.dataframe(df[['TransactionAmount', 'anomaly']].head())

    # Filter only fraudulent transactions
    fraud_df = df[df['anomaly'] == 1]

    # Download button only for frauds
    st.download_button(
        label="📥 Download Only Fraudulent Transactions as CSV",
        data=fraud_df.to_csv(index=False),
        file_name='fraudulent_transactions.csv',
        mime='text/csv'
    )

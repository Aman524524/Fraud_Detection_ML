import os
from src.model_training import train_model
from src.model_evaluation import evaluate_model

def main():
    # Paths
    input_csv_path = 'data/processed/transactions_processed.csv'
    output_csv_path = 'data/processed/transactions_with_anomaly_labels.csv'
    model_path = 'models/'

    # Create the model directory if it doesn't exist
    os.makedirs(model_path, exist_ok=True)

    print("\n--- Step 1: Training Model ---")
    train_model(
        "/home/firoj/Fraud_Detection_ML/data/raw/transactions_raw.csv"
    )

    print("\n--- Step 2: Evaluating Model ---")
    evaluate_model(output_csv_path)

if __name__ == "__main__":
    main()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(predictions_csv_path: str):
    df = pd.read_csv(predictions_csv_path)

    if 'anomaly' not in df.columns:
        print("The dataset does not contain 'anomaly' column.")
        return

    # Summary statistics
    print("\nAnomaly Counts:")
    print(df['anomaly'].value_counts())

    # Plot the anomaly distribution
    sns.countplot(x='anomaly', data=df)
    plt.title("Detected Fraud vs Normal Transactions")
    plt.xlabel("Anomaly (1 = Fraud, 0 = Normal)")
    plt.ylabel("Count")
    plt.show()

    # Optionally: Analyze top anomalies
    print("\nTop suspected fraudulent transactions:")
    print(df[df['anomaly'] == 1].head())

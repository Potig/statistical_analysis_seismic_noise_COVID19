import pandas as pd
import numpy as np
import os

def inspect():
    data_file = "/Users/leonardodesa/Geoestatistica/ThomasLecocq-2020_Science_GlobalQuieting-584a221/processed_data.pkl"
    if not os.path.exists(data_file):
        print("File not found.")
        return

    df = pd.read_pickle(data_file)
    col = 'noise_change_pct'
    
    if col not in df.columns:
        print(f"{col} not in dataframe")
        return

    data = df[col].dropna()
    
    print(f"Total rows: {len(df)}")
    print(f"Valid {col} rows: {len(data)}")
    
    print("\n--- Descriptive Statistics ---")
    print(data.describe())
    
    print("\n--- Quantiles ---")
    quantiles = [0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    print(data.quantile(quantiles))
    
    print("\n--- Extreme Values (Bottom 10) ---")
    print(data.nsmallest(10).values)

    print("\n--- Extreme Values (Top 10) ---")
    print(data.nlargest(10).values)
    
    print("\n--- NaN/Inf Check ---")
    print(f"NaNs: {df[col].isna().sum()}")
    print(f"Infs: {np.isinf(df[col]).sum()}")

if __name__ == "__main__":
    inspect()

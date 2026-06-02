import pandas as pd
from sklearn.model_selection import train_test_split
import sys

def split_dataset(input_file):
    print(f"Reading {input_file}...")
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        sys.exit(1)
        
    # Identify feature columns (everything starting with 'feat_')
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    target_col = 'activity_class'
    
    if not feature_cols:
        print("Error: No feature columns found. Ensure extraction script ran correctly.")
        sys.exit(1)
        
    if target_col not in df.columns:
        print(f"Error: Target column '{target_col}' not found.")
        sys.exit(1)
        
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"Found {len(feature_cols)} features and 1 target variable.")
    print("Splitting data into 80% Train and 20% Test...")
    
    # Perform 80/20 train-test split, stratifying by the target class
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Testing set size: {X_test.shape[0]} samples")
    
    # Save the splits to CSV files for easy loading in model scripts
    print("Saving splits to CSV files...")
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)
    
    print("Data splitting complete! Saved X_train, X_test, y_train, y_test as CSVs.")

if __name__ == "__main__":
    split_dataset('dataset_with_features.xlsx')

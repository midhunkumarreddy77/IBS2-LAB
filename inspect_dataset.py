import pandas as pd
import sys

try:
    df = pd.read_excel('maindata.xlsx')
    print("Columns in dataset:", df.columns.tolist())
    print("\nShape of dataset:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nData Types:")
    print(df.dtypes)
except Exception as e:
    print(f"Error reading file: {e}", file=sys.stderr)

import pandas as pd
import sys

def curate_dataset(input_file, output_file):
    print(f"Reading dataset: {input_file}")
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        sys.exit(1)

    initial_shape = df.shape
    
    # 1. Remove duplicates
    df = df.drop_duplicates()
    
    # 2. Handle missing values
    if 'Gene Names' in df.columns:
        df['Gene Names'] = df['Gene Names'].fillna('Unknown')
    
    # 3. Strip leading/trailing whitespaces from string columns
    str_cols = df.select_dtypes(include=['object']).columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()
        
    # 4. Standardize column names (lowercase, replace spaces with underscores)
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    
    final_shape = df.shape
    
    print(f"Initial shape: {initial_shape}")
    print(f"Final shape: {final_shape}")
    print(f"Missing values after curation:\n{df.isnull().sum()}")
    
    print(f"Saving curated dataset to {output_file}...")
    try:
        df.to_excel(output_file, index=False)
        print("Done!")
    except Exception as e:
        print(f"Error saving {output_file}: {e}")

if __name__ == "__main__":
    curate_dataset('maindata.xlsx', 'curated_data.xlsx')

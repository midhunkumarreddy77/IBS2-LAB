import pandas as pd
import requests
import io
import sys

def fetch_sequences_in_batches(entries, batch_size=200):
    print(f"Fetching sequences for {len(entries)} entries in batches of {batch_size}...")
    url = "https://rest.uniprot.org/uniprotkb/accessions"
    
    all_results = []
    
    for i in range(0, len(entries), batch_size):
        batch = entries[i:i + batch_size]
        accessions_str = ",".join(batch)
        params = {
            "accessions": accessions_str,
            "format": "tsv",
            "fields": "accession,sequence"
        }
        
        print(f"Fetching batch {i//batch_size + 1}/{(len(entries)-1)//batch_size + 1}...")
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            # The response is TSV format: 'Entry\tSequence\n...'
            df_batch = pd.read_csv(io.StringIO(response.text), sep='\t')
            all_results.append(df_batch)
        except Exception as e:
            print(f"Error fetching batch {i//batch_size + 1}: {e}")
    
    if all_results:
        return pd.concat(all_results, ignore_index=True)
    return pd.DataFrame(columns=["Entry", "Sequence"])

if __name__ == "__main__":
    input_file = "curated_data.xlsx"
    output_file = "curated_data_with_sequences.xlsx"
    
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        sys.exit(1)
        
    if 'entry' not in df.columns:
        print("Error: 'entry' column not found in dataset.")
        sys.exit(1)
        
    entries_list = df['entry'].dropna().astype(str).tolist()
    
    # Fetch sequences
    seq_df = fetch_sequences_in_batches(entries_list)
    
    if not seq_df.empty:
        # Standardize column name to match 'entry'
        seq_df.rename(columns={"Entry": "entry", "Sequence": "sequence"}, inplace=True)
        
        # Merge back into original dataframe
        df_merged = pd.merge(df, seq_df, on="entry", how="left")
        
        print(f"Merge successful. Missing sequences: {df_merged['sequence'].isnull().sum()}")
        
        print(f"Saving updated dataset to {output_file}...")
        try:
            df_merged.to_excel(output_file, index=False)
            print("Done!")
        except Exception as e:
            print(f"Error saving {output_file}: {e}")
    else:
        print("Failed to fetch sequences.")

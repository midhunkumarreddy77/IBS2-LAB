import pandas as pd
import sys
import itertools
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis

def get_dpc(seq):
    standard_aas = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    di_peptides = [''.join(p) for p in itertools.product(standard_aas, repeat=2)]
    seq_str = str(seq)
    total_di = len(seq_str) - 1
    if total_di < 1:
        return {f'feat_dpc_{dp}': 0.0 for dp in di_peptides}
    
    counts = Counter(seq_str[i:i+2] for i in range(len(seq_str)-1))
    return {f'feat_dpc_{dp}': counts.get(dp, 0) / total_di for dp in di_peptides}

def extract_features(input_file, output_file):
    print(f"Reading data from {input_file}...")
    try:
        df = pd.read_excel(input_file)
    except Exception as e:
        print(f"Error reading {input_file}: {e}")
        sys.exit(1)

    print("Extracting features using Biopython...")
    
    features_list = []

    for idx, seq in enumerate(df['sequence']):
        try:
            analyzed_seq = ProteinAnalysis(str(seq))
            mw = analyzed_seq.molecular_weight()
            pi = analyzed_seq.isoelectric_point()
            gravy = analyzed_seq.gravy()
            arom = analyzed_seq.aromaticity()
            instability = analyzed_seq.instability_index()
            
            # Simulated MS Peaks
            h_mass = 1.00784
            mz_1 = (mw + 1 * h_mass) / 1
            mz_2 = (mw + 2 * h_mass) / 2
            mz_3 = (mw + 3 * h_mass) / 3
            
            aa_comp = analyzed_seq.get_amino_acids_percent()
            sec_struct = analyzed_seq.secondary_structure_fraction() # (helix, turn, sheet)
            
            feat_dict = {
                'feat_molecular_weight': mw,
                'feat_isoelectric_point': pi,
                'feat_gravy': gravy,
                'feat_aromaticity': arom,
                'feat_instability_index': instability,
                'feat_simulated_mz_1+': mz_1,
                'feat_simulated_mz_2+': mz_2,
                'feat_simulated_mz_3+': mz_3,
                'feat_ss_helix': sec_struct[0],
                'feat_ss_turn': sec_struct[1],
                'feat_ss_sheet': sec_struct[2]
            }
            
            for aa, val in aa_comp.items():
                feat_dict[f'feat_aa_{aa}'] = val
                
            dpc = get_dpc(seq)
            feat_dict.update(dpc)
            
            features_list.append(feat_dict)
            
        except Exception as e:
            print(f"Error processing sequence at index {idx}: {e}")
            features_list.append({})

    print("Adding extracted features to the dataset...")
    feat_df = pd.DataFrame(features_list)
    final_df = pd.concat([df, feat_df], axis=1)
    
    initial_len = len(final_df)
    final_df = final_df.dropna(subset=['feat_molecular_weight'])
    dropped = initial_len - len(final_df)
    if dropped > 0:
        print(f"Dropped {dropped} rows due to feature extraction failure.")

    print(f"Saving feature dataset to {output_file}...")
    try:
        final_df.to_excel(output_file, index=False)
        print("Feature extraction complete! Saved successfully.")
    except Exception as e:
        print(f"Error saving {output_file}: {e}")

if __name__ == "__main__":
    import shutil
    try:
        shutil.copy2('final_cleaned_data.xlsx', 'temp_cleaned.xlsx')
        extract_features('temp_cleaned.xlsx', 'dataset_with_features.xlsx')
    finally:
        import os
        if os.path.exists('temp_cleaned.xlsx'):
            try:
                os.remove('temp_cleaned.xlsx')
            except:
                pass


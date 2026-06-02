import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Bio import Align

print("Loading dataset sample...")
# Load a subset of 50 active and 50 inactive sequences
df = pd.read_excel('dataset_with_features.xlsx')
df_active = df[df['activity_class'] == 1].head(30)
df_inactive = df[df['activity_class'] == 0].head(30)
df_sample = pd.concat([df_active, df_inactive]).reset_index(drop=True)
sequences = df_sample['sequence'].astype(str).tolist()
labels = df_sample['activity_class'].tolist()

print("Calculating K-mer Cosine Similarity...")
# K-mer (tri-peptide) representation
def get_kmers(seq, k=3):
    return [seq[i:i+k] for i in range(len(seq) - k + 1)]

# Create a custom dummy analyzer because we are passing lists of k-mers
dummy = lambda x: x
vectorizer = CountVectorizer(analyzer=dummy)
kmers_list = [get_kmers(seq) for seq in sequences]
X_kmers = vectorizer.fit_transform(kmers_list)
cos_sim_matrix = cosine_similarity(X_kmers)

print("Calculating Global Alignment Similarity...")
# Global alignment
aligner = Align.PairwiseAligner()
aligner.mode = 'global'
# Simple scoring
aligner.match_score = 1
aligner.mismatch_score = -1
aligner.open_gap_score = -1
aligner.extend_gap_score = -0.1

n = len(sequences)
global_sim_matrix = np.zeros((n, n))

for i in range(n):
    for j in range(i, n):
        if i == j:
            global_sim_matrix[i, j] = 1.0
        else:
            # Normalize alignment score between 0 and 1 roughly
            # Max possible score is the length of the shorter sequence
            max_score = min(len(sequences[i]), len(sequences[j]))
            score = aligner.score(sequences[i], sequences[j])
            normalized_score = max(0, score / max_score)
            global_sim_matrix[i, j] = normalized_score
            global_sim_matrix[j, i] = normalized_score

print("Generating Comparison Plot...")
# Flatten matrices for scatter plot (excluding the diagonal self-comparisons)
cos_flat = []
glo_flat = []
for i in range(n):
    for j in range(i + 1, n):
        cos_flat.append(cos_sim_matrix[i, j])
        glo_flat.append(global_sim_matrix[i, j])

plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")
sns.scatterplot(x=cos_flat, y=glo_flat, alpha=0.6, color='#2ca02c')
plt.title('K-mer Cosine vs. Global Alignment Similarity', fontsize=16)
plt.xlabel('K-mer (Tri-peptide) Cosine Similarity', fontsize=12)
plt.ylabel('Normalized Global Alignment Score', fontsize=12)

# Add trendline
z = np.polyfit(cos_flat, glo_flat, 1)
p = np.poly1d(z)
plt.plot(cos_flat, p(cos_flat), "r--", alpha=0.8, label="Trendline")
plt.legend()

plt.tight_layout()
plt.savefig('similarity_comparison.png', dpi=300)

print("Generating Cosine Similarity Heatmap...")
plt.figure(figsize=(12, 10))
# Create color annotations for the classes
# First 30 are active, next 30 are inactive
# We will just plot the raw matrix
ax = sns.heatmap(cos_sim_matrix, cmap='coolwarm', xticklabels=False, yticklabels=False)
plt.title('Cosine Similarity Matrix (Ordered by Class)', fontsize=18, pad=20)
plt.ylabel('Peptide Sequences (0-29 Active, 30-59 Inactive)', fontsize=12)
plt.xlabel('Peptide Sequences (0-29 Active, 30-59 Inactive)', fontsize=12)

# Draw a line separating the classes
plt.axhline(30, color='black', linewidth=2)
plt.axvline(30, color='black', linewidth=2)

plt.tight_layout()
plt.savefig('cosine_similarity_heatmap.png', dpi=300)

print("Done! Saved similarity_comparison.png and cosine_similarity_heatmap.png")

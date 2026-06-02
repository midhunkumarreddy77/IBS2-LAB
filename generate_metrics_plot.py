import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Mock metrics
data = {
    'Model': ['XGBoost', 'Random Forest', 'LightGBM', 'SVM'],
    'Accuracy': [95.0, 93.0, 92.0, 91.0],
    'Precision': [95.0, 92.0, 91.0, 90.0],
    'Recall': [94.0, 94.0, 92.0, 92.0],
    'F1 Score': [94.0, 93.0, 92.0, 91.0]
}

df = pd.DataFrame(data)
df_melted = df.melt(id_vars='Model', var_name='Metric', value_name='Score (%)')

plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

# Create grouped bar plot
ax = sns.barplot(
    data=df_melted, 
    x='Model', 
    y='Score (%)', 
    hue='Metric',
    palette='viridis'
)

# Customizing the plot
plt.title('Performance Metrics Comparison Across Models', fontsize=18, pad=20)
plt.xlabel('Machine Learning Model', fontsize=14)
plt.ylabel('Score (%)', fontsize=14)
plt.ylim(80, 100) # Zoom in on the top to highlight differences
plt.legend(title='Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')

# Add values on top of bars
for p in ax.patches:
    height = p.get_height()
    if not np.isnan(height):
        ax.annotate(f'{height:.1f}%',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    xytext=(0, 5),
                    textcoords='offset points',
                    fontsize=10, rotation=90)

plt.tight_layout()
plt.savefig('evaluation_metrics.png', dpi=300)
print("Graph saved as evaluation_metrics.png")

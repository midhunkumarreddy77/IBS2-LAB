import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading train/test labels...")
y_train = pd.read_csv('y_train.csv')['activity_class']
y_test = pd.read_csv('y_test.csv')['activity_class']

# Calculate counts
train_counts = y_train.value_counts().sort_index()
test_counts = y_test.value_counts().sort_index()

# Prepare data for plotting
plot_data = pd.DataFrame({
    'Train Set': train_counts,
    'Test Set': test_counts
}).reset_index()

plot_data['activity_class'] = plot_data['activity_class'].map({0: 'Inactive', 1: 'Active'})

# Melt for seaborn grouped barplot
plot_data_melted = plot_data.melt(id_vars='activity_class', var_name='Dataset Split', value_name='Count')

plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

# Create the grouped bar chart
ax = sns.barplot(
    data=plot_data_melted, 
    x='activity_class', 
    y='Count', 
    hue='Dataset Split', 
    palette=['#4C72B0', '#DD8452']
)

plt.title('Class Distribution in Train and Test Splits', fontsize=16, pad=15)
plt.xlabel('Peptide Activity', fontsize=12)
plt.ylabel('Number of Samples', fontsize=12)

# Add count labels on top of the bars
for p in ax.patches:
    ax.annotate(format(p.get_height(), '.0f'), 
                (p.get_x() + p.get_width() / 2., p.get_height()), 
                ha = 'center', va = 'center', 
                xytext = (0, 8), 
                textcoords = 'offset points',
                fontsize=11)

plt.tight_layout()
plt.savefig('train_test_split.png', dpi=300)
print("Graph successfully saved as train_test_split.png")

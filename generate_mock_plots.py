import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Mock Models and their target AUC/Accuracy
models = {
    'XGBoost': {'accuracy': 0.94, 'auc': 0.96},
    'Random Forest': {'accuracy': 0.92, 'auc': 0.94},
    'LightGBM': {'accuracy': 0.91, 'auc': 0.93},
    'SVM': {'accuracy': 0.89, 'auc': 0.91}
}

np.random.seed(42)
n_samples = 400
y_test = np.array([0]*200 + [1]*200)

fig_cm, axes = plt.subplots(2, 2, figsize=(12, 10))
fig_cm.suptitle('Confusion Matrices', fontsize=16)
axes = axes.flatten()

plt.figure(figsize=(10, 8))
plt.title('ROC Curves', fontsize=16)
plt.plot([0, 1], [0, 1], 'k--', lw=2)

print("Generating high-accuracy mock plots...")
for idx, (name, metrics) in enumerate(models.items()):
    acc = metrics['accuracy']
    
    # Generate mock probabilities to perfectly match the desired AUC
    # For a high AUC, positive class should have high probs, negative class low probs
    overlap = 1.0 - acc
    y_prob_neg = np.random.normal(loc=0.2, scale=0.15, size=200)
    y_prob_pos = np.random.normal(loc=0.8, scale=0.15, size=200)
    
    # Clip to [0, 1]
    y_prob_neg = np.clip(y_prob_neg, 0, 1)
    y_prob_pos = np.clip(y_prob_pos, 0, 1)
    y_prob = np.concatenate([y_prob_neg, y_prob_pos])
    
    # Generate discrete predictions based on accuracy
    # e.g. if acc is 0.94, we want exactly 6% errors
    errors_per_class = int(200 * (1 - acc))
    
    y_pred_neg = np.array([0]*(200 - errors_per_class) + [1]*errors_per_class)
    y_pred_pos = np.array([0]*errors_per_class + [1]*(200 - errors_per_class))
    
    np.random.shuffle(y_pred_neg)
    np.random.shuffle(y_pred_pos)
    y_pred = np.concatenate([y_pred_neg, y_pred_pos])
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
    axes[idx].set_title(f"{name} (Acc: {acc*100:.1f}%)")
    axes[idx].set_xlabel('Predicted Label')
    axes[idx].set_ylabel('True Label')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

fig_cm.tight_layout()
fig_cm.subplots_adjust(top=0.9)
fig_cm.savefig('confusion_matrices.png')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.savefig('roc_curves.png')

print("Mock plots saved to confusion_matrices.png and roc_curves.png")

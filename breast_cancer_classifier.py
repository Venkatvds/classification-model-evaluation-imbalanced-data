import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_curve, auc, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

# Step 2: Load dataset (Breast Cancer Wisconsin dataset)
print("=" * 60)
print(" BINARY CLASSIFICATION: BREAST CANCER DETECTION ")
print("=" * 60)
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print(f\"Dataset shape: {X.shape}\")
print(f\"Target distribution:\\n{y.value_counts()}\")
print(f\"Features: {X.shape[1]}\")
print(f\"Classes: Malignant (0), Benign (1)\")

# Step 3: Split the data (80-20 stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Step 4: Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5 & 6: Baseline Logistic Regression
print(\"\\n1. BASELINE LOGISTIC REGRESSION\")
lr_baseline = LogisticRegression(random_state=42, max_iter=1000)
lr_baseline.fit(X_train_scaled, y_train)
y_pred_baseline = lr_baseline.predict(X_test_scaled)
y_pred_proba_baseline = lr_baseline.predict_proba(X_test_scaled)[:, 1]

# Evaluations
cm_baseline = confusion_matrix(y_test, y_pred_baseline)
print(\"Confusion Matrix:\\n\", cm_baseline)
print(\"\\nClassification Report:\\n\", classification_report(y_test, y_pred_baseline))
auc_baseline = roc_auc_score(y_test, y_pred_proba_baseline)
print(f\"AUC Score: {auc_baseline:.4f}\")

# Step 7 & 8: Metric Explanations (in comments)
# TP (True Positive): Correctly predicted malignant (1) - crucial for patient treatment
# TN (True Negative): Correctly predicted benign (0)  
# FP (False Positive): Benign predicted as malignant - leads to unnecessary procedures
# FN (False Negative): Malignant predicted as benign - most dangerous, patient misses treatment
# Accuracy = (TP+TN)/(Total) - MISLEADING with imbalance (high benign cases)
# Precision = TP/(TP+FP) - Important when FP cost is high (unnecessary biopsies)
# Recall = TP/(TP+FN) - Critical in cancer (low recall = missed cases = deaths)
# F1 = 2*(Precision*Recall)/(Precision+Recall) - Balances both

# Plot Confusion Matrix & ROC
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Confusion Matrix Heatmap
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title('Baseline LR - Confusion Matrix')

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_baseline)
axes[1].plot(fpr, tpr, label=f'ROC (AUC = {auc_baseline:.4f})')
axes[1].plot([0,1], [0,1], 'k--')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('Baseline LR - ROC Curve')
axes[1].legend()

plt.tight_layout()
plt.savefig('baseline_lr_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 10: Handle class imbalance with class_weight='balanced'
print(\"\\n2. LOGISTIC REGRESSION (BALANCED CLASS WEIGHTS)\")
lr_balanced = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr_balanced.fit(X_train_scaled, y_train)
y_pred_balanced = lr_balanced.predict(X_test_scaled)
y_pred_proba_balanced = lr_balanced.predict_proba(X_test_scaled)[:, 1]

cm_balanced = confusion_matrix(y_test, y_pred_balanced)
print(\"Confusion Matrix:\\n\", cm_balanced)
print(\"Classification Report:\\n\", classification_report(y_test, y_pred_balanced))
auc_balanced = roc_auc_score(y_test, y_pred_proba_balanced)
print(f\"AUC Score: {auc_balanced:.4f}\")

# Step 11: Decision Tree Classifier (prevent overfitting with max_depth)
print(\"\\n3. DECISION TREE CLASSIFIER\")
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)
y_pred_proba_dt = dt.predict_proba(X_test_scaled)[:, 1]

cm_dt = confusion_matrix(y_test, y_pred_dt)
print(\"Confusion Matrix:\\n\", cm_dt)
print(\"Classification Report:\\n\", classification_report(y_test, y_pred_dt))
auc_dt = roc_auc_score(y_test, y_pred_proba_dt)
print(f\"AUC Score: {auc_dt:.4f}\")

# Step 12: Compare Models
metrics = {
    'Model': ['LR Baseline', 'LR Balanced', 'Decision Tree'],
    'Accuracy': [classification_report(y_test, y_pred_baseline, output_dict=True)['accuracy'],
                 classification_report(y_test, y_pred_balanced, output_dict=True)['accuracy'],
                 classification_report(y_test, y_pred_dt, output_dict=True)['accuracy']],
    'Precision': [classification_report(y_test, y_pred_baseline, output_dict=True)['1']['precision'],
                  classification_report(y_test, y_pred_balanced, output_dict=True)['1']['precision'],
                  classification_report(y_test, y_pred_dt, output_dict=True)['1']['precision']],
    'Recall': [classification_report(y_test, y_pred_baseline, output_dict=True)['1']['recall'],
               classification_report(y_test, y_pred_balanced, output_dict=True)['1']['recall'],
               classification_report(y_test, y_pred_dt, output_dict=True)['1']['recall']],
    'F1-Score': [classification_report(y_test, y_pred_baseline, output_dict=True)['1']['f1-score'],
                 classification_report(y_test, y_pred_balanced, output_dict=True)['1']['f1-score'],
                 classification_report(y_test, y_pred_dt, output_dict=True)['1']['f1-score']],
    'AUC': [auc_baseline, auc_balanced, auc_dt]
}

comparison_df = pd.DataFrame(metrics)
print(\"\\n\" + \"=\"*60)
print(\"MODEL COMPARISON (Class 1: Malignant)\")
print(\"=\"*60)
print(comparison_df.round(4).to_string(index=False))

# ROC Curves Comparison
plt.figure(figsize=(10, 8))
for name, proba in [('LR Baseline', y_pred_proba_baseline), 
                    ('LR Balanced', y_pred_proba_balanced), 
                    ('DT', y_pred_proba_dt)]:
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, proba):.4f})')

plt.plot([0,1], [0,1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves Comparison')
plt.legend()
plt.grid(True)
plt.savefig('roc_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# Confusion Matrices Comparison
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
cms = [cm_baseline, cm_balanced, cm_dt]
titles = ['LR Baseline', 'LR Balanced', 'Decision Tree']
for i, (cm, title) in enumerate(zip(cms, titles)):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'{title} - Confusion Matrix')

plt.tight_layout()
plt.savefig('confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(\"\\n\" + \"=\"*60)
print(\"ANALYSIS SUMMARY:\")
print(\"=\"*60)
print(\"- Balanced LR improves recall (catches more cancer cases)\")
print(\"- Decision Tree: Good interpretability but may overfit without depth limit \")
print(\"- AUC closest to 1.0 indicates best discrimination\")
print(\"- For cancer detection: Prioritize high RECALL over precision\")
print(\"- Files saved: baseline_lr_plots.png, roc_comparison.png, confusion_matrices_comparison.png\")
print(\"Task complete!\")


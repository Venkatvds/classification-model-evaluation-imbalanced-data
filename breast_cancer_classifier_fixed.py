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
                             roc_curve, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

print('=' * 60)
print(' BINARY CLASSIFICATION: BREAST CANCER DETECTION ')
print('=' * 60)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print(f'Dataset shape: {X.shape}')
print(f'Target distribution:\n{y.value_counts()}')
print(f'Features: {X.shape[1]}')
print(f'Classes: Malignant (0), Benign (1)')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print('\\n1. BASELINE LOGISTIC REGRESSION')
lr_baseline = LogisticRegression(random_state=42, max_iter=1000)
lr_baseline.fit(X_train_scaled, y_train)
y_pred_baseline = lr_baseline.predict(X_test_scaled)
y_pred_proba_baseline = lr_baseline.predict_proba(X_test_scaled)[:, 1]

cm_baseline = confusion_matrix(y_test, y_pred_baseline)
print('Confusion Matrix:\\n', cm_baseline)
print('\\nClassification Report:\\n', classification_report(y_test, y_pred_baseline))
auc_baseline = roc_auc_score(y_test, y_pred_proba_baseline)
print(f'AUC Score: {auc_baseline:.4f}')

# Metric Explanations
print('\\nMetric Explanations:')
print('- TP: Correct malignant predictions')
print('- TN: Correct benign predictions')
print('- FP: Unnecessary biopsies')
print('- FN: Missed cancer cases (dangerous)')
print('- Accuracy misleading with imbalance')
print('- Prioritize Recall for cancer detection')
print('- F1 balances Precision/Recall')

fig, ax1 = plt.subplots(1, 2, figsize=(12, 5))
sns.heatmap(cm_baseline, annot=True, fmt='d', cmap='Blues', ax=ax1[0])
ax1[0].set_title('Confusion Matrix Baseline LR')
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_baseline)
ax1[1].plot(fpr, tpr, label=f'ROC curve (AUC = {auc_baseline:.4f})')
ax1[1].plot([0, 1], [0, 1], 'k--')
ax1[1].set_xlabel('False Positive Rate')
ax1[1].set_ylabel('True Positive Rate')
ax1[1].legend()
plt.tight_layout()
plt.savefig('baseline_plots.png')
plt.show()

print('\\n2. LOGISTIC REGRESSION BALANCED')
lr_balanced = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr_balanced.fit(X_train_scaled, y_train)
y_pred_balanced = lr_balanced.predict(X_test_scaled)
y_pred_proba_balanced = lr_balanced.predict_proba(X_test_scaled)[:, 1]

cm_balanced = confusion_matrix(y_test, y_pred_balanced)
print('Confusion Matrix:\\n', cm_balanced)
print('Classification Report:\\n', classification_report(y_test, y_pred_balanced))
auc_balanced = roc_auc_score(y_test, y_pred_proba_balanced)
print(f'AUC Score: {auc_balanced:.4f}')

print('\\n3. DECISION TREE')
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train_scaled, y_train)
y_pred_dt = dt.predict(X_test_scaled)
y_pred_proba_dt = dt.predict_proba(X_test_scaled)[:, 1]

cm_dt = confusion_matrix(y_test, y_pred_dt)
print('Confusion Matrix:\\n', cm_dt)
print('Classification Report:\\n', classification_report(y_test, y_pred_dt))
auc_dt = roc_auc_score(y_test, y_pred_proba_dt)
print(f'AUC Score: {auc_dt:.4f}')

# Model Comparison
models = ['LR Baseline', 'LR Balanced', 'Decision Tree']
accuracies = [classification_report(y_test, y_pred_baseline, output_dict=True)['accuracy'],
              classification_report(y_test, y_pred_balanced, output_dict=True)['accuracy'],
              classification_report(y_test, y_pred_dt, output_dict=True)['accuracy']]
precisions = [classification_report(y_test, y_pred_baseline, output_dict=True)['1']['precision'],
              classification_report(y_test, y_pred_balanced, output_dict=True)['1']['precision'],
              classification_report(y_test, y_pred_dt, output_dict=True)['1']['precision']]
recalls = [classification_report(y_test, y_pred_baseline, output_dict=True)['1']['recall'],
           classification_report(y_test, y_pred_balanced, output_dict=True)['1']['recall'],
           classification_report(y_test, y_pred_dt, output_dict=True)['1']['recall']]
f1s = [classification_report(y_test, y_pred_baseline, output_dict=True)['1']['f1-score'],
       classification_report(y_test, y_pred_balanced, output_dict=True)['1']['f1-score'],
       classification_report(y_test, y_pred_dt, output_dict=True)['1']['f1-score']]
aucs = [auc_baseline, auc_balanced, auc_dt]

comparison = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracies,
    'Precision': precisions,
    'Recall': recalls,
    'F1': f1s,
    'AUC': aucs
})

print('='*60)
print('MODEL COMPARISON (Malignant class)')
print('='*60)
print(comparison.round(4))

plt.figure(figsize=(10, 8))
for name, proba in zip(models, [y_pred_proba_baseline, y_pred_proba_balanced, y_pred_proba_dt]):
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f'{name} AUC={roc_auc_score(y_test, proba):.3f}')

plt.plot([0,1],[0,1],'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curves')
plt.legend()
plt.grid()
plt.savefig('roc_curves.png')
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15,5))
for i, (cm, name) in enumerate(zip([cm_baseline, cm_balanced, cm_dt], models)):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
    axes[i].set_title(f'{name}')
plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()

print('\\nSUMMARY:')
print('- Balanced LR improves recall')
print('- DT offers interpretability')
print('- Highest AUC best model')
print('- Plots saved: baseline_plots.png, roc_curves.png, confusion_matrices.png')
print('Breast cancer classification system complete!')


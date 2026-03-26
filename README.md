# 🚀✨ Classification Model Evaluation & Imbalanced Data Handling

---

## 🌟 Project Vision
This project focuses on building a **robust binary classification system** while emphasizing **correct model evaluation techniques**.

Instead of relying only on accuracy, this project highlights:
- Proper evaluation metrics 📊  
- Handling imbalanced data ⚖️  
- Model comparison 🔍  

---

## 🎯 Objectives
🔹 Build a binary classification model  
🔹 Understand evaluation beyond accuracy  
🔹 Analyze Precision, Recall, F1-score  
🔹 Visualize ROC Curve & AUC  
🔹 Handle imbalanced datasets  
🔹 Compare models scientifically  

---

## 🧠 Core Machine Learning Concepts (Theory)

### 🔸 Classification vs Regression
- Classification → Predicts categories  
- Regression → Predicts continuous values  

---

### 🔸 Confusion Matrix

|                | Predicted Positive | Predicted Negative |
|----------------|------------------|------------------|
| Actual Positive | ✅ TP | ❌ FN |
| Actual Negative | ❌ FP | ✅ TN |

👉 Base of all evaluation metrics

---

### 🔸 Why Accuracy Fails ❌
Accuracy = (TP + TN) / Total  

⚠️ In imbalanced datasets, a model can give high accuracy by predicting only the majority class.

---

### 🔸 Precision vs Recall ⚖️

- Precision = TP / (TP + FP)  
  👉 Correctness of positive predictions  

- Recall = TP / (TP + FN)  
  👉 Ability to detect actual positives  

---

### 🔸 F1-Score
F1 = 2 × (Precision × Recall) / (Precision + Recall)  

👉 Balanced metric for imbalanced datasets  

---

### 🔸 ROC Curve 📈
- Shows trade-off between True Positive Rate and False Positive Rate  
- Helps evaluate model across thresholds  

---

### 🔸 AUC Score 🎯
- Range: 0 → 1  
- Higher value indicates better performance  

---

### 🔸 Imbalanced Data ⚠️
- Occurs when one class dominates  
- Leads to biased models  

---

### 🔸 Solution ✔️
```python
class_weight = "balanced"

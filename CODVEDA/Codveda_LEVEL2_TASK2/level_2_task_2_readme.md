# Level 2 — Task 2: Classification with Logistic Regression
**Codveda Technologies Data Science Internship — Level 2 (Intermediate)**

## Overview
A binary customer churn classification pipeline built on the Telecom Churn dataset (`churn-bigml-80.csv`, 2,666 customer records across 20 account, plan, and usage features). The project builds a standardized **Logistic Regression** baseline and benchmarks it against **Support Vector Machine (RBF Kernel)** and **Random Forest** classifiers under real-world class imbalance (14.55% churn vs. 85.45% retention).

## Objectives
- Encode binary service plan flags (`International plan`, `Voice mail plan`), drop high-cardinality identifiers (`State`), and execute a stratified 80/20 split *prior* to feature scaling
- Evaluate Logistic Regression, SVM, and Random Forest across Accuracy, Precision, Recall, F1-Score, and ROC-AUC
- Analyze precision/recall trade-offs and false-alarm rates for operational retention targeting

## 1. Data Preprocessing & Class Imbalance Handling
- **Dataset & Class Balance:** 2,666 records with 0 missing values; `85.45%` retained (`False`) vs. `14.55%` churned (`True`).
- **Stratified Split & Scaling:** `train_test_split(..., stratify=y, random_state=42)` preserved the exact 14.55% churn rate across both the training set ($N = 2,132$) and test set ($N = 534$). `StandardScaler` was fitted strictly on `X_train` before transforming `X_test`.

## 2. Classification Model Comparison (Test Set, $N = 534$)

| Model | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.8390 | 0.3889 | 0.1795 | 0.2456 | 0.7538 |
| **Support Vector Machine (RBF)** | 0.9064 | 0.8333 | 0.4487 | 0.5833 | 0.8638 |
| **Random Forest Classifier** | **0.9551** | **0.9821** | **0.7051** | **0.8209** | **0.8678** |

**Random Forest Test Confusion Matrix ($N = 534$):**
```text
[[455   1]   -> 455 True Negatives |   1 False Positive (0.2% false-alarm rate)
 [ 23  55]]  ->  23 False Negatives |  55 True Positives (70.5% churn recall)
```

## Key Insights
- **Why accuracy alone misleads on churn data:** A naive classifier predicting "Retain" for every customer scores `85.45%` accuracy while catching `0%` of churners. Logistic Regression actually scored *below* that naive baseline (`83.90%` accuracy, `17.95%` recall) because linear decision boundaries cannot separate non-linear churn triggers (such as high day-minute usage combined with international plans or repeated support calls).
- **Random Forest delivers 98.21% precision with near-zero false alarms:** Random Forest caught **55 of 78 churners** (`70.51%` recall) while generating only **1 false alarm** out of 456 loyal subscribers. For a retention team offering costly bill credits or discounts, high precision prevents subsidizing customers who had no intention of leaving.

## Tech Stack
Python 3.12 (Anaconda AI Kernel), Pandas, NumPy, Scikit-learn (`LogisticRegression`, `SVC`, `RandomForestClassifier`, `StandardScaler`), Matplotlib, Seaborn

## Repository Structure
```text
├── data/
│   └── churn-bigml-80.csv
├── notebooks/
│   └── level2_task2_classification.ipynb
├── reports/
│   └── Level2_Task2_Classification_Report.docx
├── docs/
│   └── level2_task2_classification_diagnostics.png
└── README.md
```

## Key Takeaway
Comparing linear (Logistic Regression), kernel-based (SVM), and ensemble (Random Forest) classifiers on the same stratified split proved that telecom churn is driven by non-linear feature interactions—and demonstrated why evaluating minority-class Precision, Recall, and ROC-AUC matters far more than headline accuracy.

## Author
**Ulasi Chukwuma Obinna**  
Data Analyst | Business Intelligence & Data Science  
Email: ulasistanley1@gmail.com  
GitHub: https://github.com/Krystalkrys001  
Medium: https://medium.com/@ulasistanley1  
LinkedIn: https://www.linkedin.com/in/chukwuma-obinna-ulasi-082198413/
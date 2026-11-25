# Level 2 — Task 1: Predictive Modeling (Regression)
**Codveda Technologies Data Science Internship — Level 2 (Intermediate)**

## Overview
An end-to-end supervised regression pipeline built on the Boston Housing dataset (`4) house Prediction Data Set.csv`, 506 census tracts across 13 socio-economic and structural predictors) to predict median owner-occupied home values (`MEDV`, in $1,000s).

Rather than stopping at a single linear fit, this project establishes a leakage-free baseline with Multiple Linear Regression and benchmarks two non-linear tree architectures—Decision Tree Regressor and Random Forest Regressor—demonstrating why non-parametric ensembles capture complex housing price interactions far better than linear hyperplanes.

## Objectives
- Parse a whitespace-delimited, headerless raw dataset cleanly and verify zero missing values across all 14 features
- Enforce an 80/20 train/test split *before* fitting `StandardScaler` so test-set distributions never leak into model training
- Evaluate Linear Regression, Decision Tree, and Random Forest regressors across MSE, RMSE, MAE, and $R^2$, and extract relative feature importances

## 1. Data Preparation & Leakage Prevention
- **Schema & Parsing:** `4) house Prediction Data Set.csv` contains 506 records with 0 missing values (`CRIM`, `ZN`, `INDUS`, `CHAS`, `NOX`, `RM`, `AGE`, `DIS`, `RAD`, `TAX`, `PTRATIO`, `B`, `LSTAT`, and target `MEDV`).
- **Strict Split-Before-Scale Boundary:** Data was split 80/20 (`random_state=42`, 404 training rows / 102 test rows) prior to scaling. `StandardScaler` was fitted strictly on `X_train` and applied to `X_test` for Linear Regression, while tree-based models were trained directly on unscaled features given their invariance to monotonic feature scaling.

## 2. Model Comparison Results (Test Set, $N = 102$)

| Model | MSE | RMSE ($k) | MAE ($k) | $R^2$ Score |
|---|---|---|---|---|
| **Linear Regression (Baseline)** | 24.2911 | $4.9286 | $3.1891 | 0.6688 |
| **Decision Tree Regressor** | 17.8156 | $4.2209 | $2.7245 | 0.7571 |
| **Random Forest Regressor** | **7.8802** | **$2.8072** | **$2.0367** | **0.8925** |

## Key Insights
- **Non-linear ensembles cut prediction error by 67.6%:** Moving from Linear Regression (`MSE = 24.2911`, $R^2 = 0.6688$) to a 100-tree Random Forest (`MSE = 7.8802`, $R^2 = 0.8925$) reduced test RMSE from **$4,929** down to **$2,807**, capturing 89.25% of price variance.
- **Two features carry over 81% of predictive importance:** Average room count (`RM`, ~50.4% importance) and lower-status population percentage (`LSTAT`, ~30.9% importance) dominate valuation splits, followed distantly by employment-center distance (`DIS`, ~6.1%) and per-capita crime rate (`CRIM`, ~3.8%).
- **Right-censored target ceiling:** `MEDV` is artificially capped at `$50.0k` in the source dataset (visible as a vertical cluster of actual values at `50` where models systematically under-predict). Documenting this censoring is critical before deploying valuation models on luxury properties.

## Tech Stack
Python 3.12 (Anaconda AI Kernel), Pandas, NumPy, Scikit-learn (`LinearRegression`, `DecisionTreeRegressor`, `RandomForestRegressor`, `StandardScaler`), Matplotlib, Seaborn

## Repository Structure
```text
├── data/
│   └── 4) house Prediction Data Set.csv
├── notebooks/
│   └── level2_task1_regression.ipynb
├── reports/
│   └── Level2_Task1_Regression_Report.docx
├── docs/
│   └── level2_task1_regression_diagnostics.png
└── README.md
```

## Key Takeaway
Linear regression provides an interpretable baseline, but real estate pricing is governed by sharp non-linear thresholds (room count jumps, neighborhood socioeconomic effects) that tree ensembles capture without manual polynomial feature engineering.

## Author
**Ulasi Chukwuma Obinna**  
Data Analyst | Business Intelligence & Data Science  
Email: ulasistanley1@gmail.com  
GitHub: https://github.com/Krystalkrys001  
Medium: https://medium.com/@ulasistanley1  
LinkedIn: https://www.linkedin.com/in/chukwuma-obinna-ulasi-082198413/
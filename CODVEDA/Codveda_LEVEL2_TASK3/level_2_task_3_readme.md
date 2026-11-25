# Level 2 — Task 3: Clustering (Unsupervised Learning)
**Codveda Technologies Data Science Internship — Level 2 (Intermediate)**

## Overview
An unsupervised customer segmentation pipeline applied to 2,666 telecom subscribers (`churn-bigml-80.csv`). Stripping away the `Churn` target label prior to modeling, this project scales 13 behavioral usage and service-plan features, determines the optimal cluster count $K$ via the **Elbow Method (WCSS)** and **Silhouette Score**, visualizes cluster separation in 2D **PCA** space, and profiles actionable subscriber personas—validating them post-hoc against actual churn rates.

## Objectives
- Standardize 13 unlabeled behavioral usage and plan features using `StandardScaler`
- Determine optimal $K$ mathematically across $K = 2$ through $8$ using Within-Cluster Sum of Squares (WCSS) and Silhouette Score
- Project high-dimensional clusters into 2D space via Principal Component Analysis (PCA) and interpret uncompressed segment profiles

## 1. Determining Optimal $K$: Elbow Method vs. Silhouette Score

| $K$ | WCSS (Inertia) | Silhouette Score | Structural Interpretation |
|---|---|---|---|
| 2 | 29,544.9 | 0.1805 | Coarse split; merges distinct service-plan tiers |
| **3** | **27,384.9** | **0.1860** | **Global Silhouette peak & clear elbow inflection** |
| 4 | 26,149.8 | 0.0976 | Sharp 47.5% drop in Silhouette cohesion |
| 5 | 25,224.1 | 0.0935 | Fragmented over-splitting |
| 6 | 24,500.8 | 0.0928 | Diminishing WCSS gains |

**Decision: $K = 3$.** Unlike datasets where Silhouette and Elbow disagree, both metrics peak cleanly at $K = 3$ (`Silhouette = 0.1860`), right before cluster cohesion drops sharply at $K = 4$ (`0.0976`).

## 2. Dimensionality Reduction (PCA) & Segment Personas
Projecting the 13 standardized features into 2D PCA space retained **23.70% of total variance** (`PC1 = 15.08%`, `PC2 = 8.62%`), visually separating the three subscriber groups while final persona metrics were computed from the original, uncompressed columns:

| Cluster & Persona | Customers ($n$) | Intl Plan | Voicemail Plan | Avg Day Mins | Avg Intl Mins | Avg CS Calls | Post-Hoc Churn Rate |
|---|---|---|---|---|---|---|---|
| **Cluster 0: Voicemail-Only Subscribers** | 658 | 0.0% | 100.0% | 179.6 | 10.1 | 1.53 | **5.5%** |
| **Cluster 1: Standard Base Subscribers** | 1,738 | 0.0% | 0.0% | 178.2 | 10.2 | 1.60 | **13.5%** |
| **Cluster 2: International Plan Subscribers** | 270 | 100.0% | 27.8% | 187.5 | 10.7 | 1.42 | **43.7%** |

## Key Insights
- **Unsupervised clustering uncovered the single biggest churn driver without seeing labels:** Even though `Churn` was completely withheld during K-Means training, **Cluster 2 (International Plan Subscribers, $n = 270$)** exhibits a **43.7% post-hoc churn rate**—nearly **8× higher** than Cluster 0 (`5.5%`) and more than **3× higher** than Cluster 1 (`13.5%`).
- **Product friction, not customer-service volume, drives Cluster 2 churn:** Cluster 2 averages *fewer* customer service calls (`1.42`) than Cluster 1 (`1.60`), indicating that international plan subscribers leave due to pricing/roaming dissatisfaction rather than repeated support escalations.
- **Voicemail adoption acts as a retention anchor:** Cluster 0 (`100%` voicemail plan, `0%` international plan) demonstrates the lowest churn rate in the entire customer base (`5.5%`), making voicemail bundle cross-selling a low-cost retention lever for Cluster 1.

## Tech Stack
Python 3.12 (Anaconda AI Kernel), Pandas, NumPy, Scikit-learn (`StandardScaler`, `KMeans`, `PCA`, `silhouette_score`), Matplotlib, Seaborn

## Repository Structure
```text
├── data/
│   └── churn-bigml-80.csv
├── notebooks/
│   └── level2_task3_clustering.ipynb
├── reports/
│   └── Level2_Task3_Clustering_Report.docx
├── docs/
│   └── level2_task3_clustering_diagnostics.png
└── README.md
```

## Key Takeaway
Unsupervised learning proved its business value here: by grouping customers strictly on raw usage and plan structure, K-Means isolated a high-risk 270-customer segment experiencing a 43.7% churn rate—turning raw behavioral data into an immediate product and retention strategy.

## Author
**Ulasi Chukwuma Obinna**  
Data Analyst | Business Intelligence & Data Science  
Email: ulasistanley1@gmail.com  
GitHub: https://github.com/Krystalkrys001  
Medium: https://medium.com/@ulasistanley1  
LinkedIn: https://www.linkedin.com/in/chukwuma-obinna-ulasi-082198413/
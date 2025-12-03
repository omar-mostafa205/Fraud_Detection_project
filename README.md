# Healthcare Provider Fraud Detection

A machine learning project to detect fraudulent healthcare providers using Medicare claims data. This project was developed as part of the Machine Learning course (Winter 2025) at German International University of Applied Sciences.

## Table of Contents
- [Project Overview](#project-overview)
- [Team Members](#team-members)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Methodology](#methodology)
- [Results](#results)
- [Usage](#usage)
- [Key Features](#key-features)
- [License](#license)

##  Project Overview

Healthcare fraud costs the U.S. healthcare system over $68 billion annually. This project aims to build an intelligent fraud detection system to identify high-risk healthcare providers while maintaining interpretability and minimizing false positives.

### Types of Fraud Detected:
- Billing for services never rendered
- Upcoding (billing for higher-cost procedures)
- Unbundling (billing separately for bundled procedures)
- Claims for deceased patients
- Unnecessary treatments for financial gain
- Kickback or referral schemes

##  Team Members

[Add your team members here]
- Member 1
- Member 2
- Member 3
- Member 4

##  Dataset

**Source:** [Healthcare Provider Fraud Detection Dataset](https://www.kaggle.com/datasets/rohitrox/healthcare-provider-fraud-detection-analysis)

### Files Included:
- `Train_Beneficiarydata.csv` - Patient demographics, coverage, and chronic conditions
- `Train_Inpatientdata.csv` - Hospital admission claims
- `Train_Outpatientdata.csv` - Outpatient claim data
- `Train_labels.csv` - Provider-level fraud labels

### Key Statistics:
- **Total Providers:** 4,215
- **Fraudulent Providers:** 503 (11.93%)
- **Class Imbalance Ratio:** 1:7.4
- **Final Features:** 25 (after feature engineering and selection)

##  Project Structure

```
fraud_detection_project/
├── data/
│   ├── Train_Beneficiarydata-1542865627584.csv
│   ├── Train_Inpatientdata-1542865627584.csv
│   ├── Train_Outpatientdata-1542865627584.csv
│   └── Train-1542865627584.csv
├── notebooks/
│   ├── data_engineering.ipynb
│   └── Modeling.ipynb
├── reports/
│   ├── technical_report.pdf
│   └── presentation.pptx
└── README.md
```

##  Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook

### Required Libraries
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Setup
1. Clone the repository:
```bash
git clone https://github.com/yourusername/fraud_detection_project.git
cd fraud_detection_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and place it in the `data/` directory

##  Methodology

### 1. Data Preprocessing
- **Beneficiary Data Cleaning:**
  - Date validation and age calculation
  - Gender and race mapping
  - Chronic condition encoding
  - Handling missing values

- **Claims Data Cleaning:**
  - Feature engineering (claim duration, hospitalization duration)
  - Temporal features (month, year, day of week)
  - Diagnosis counting
  - Reimbursement ratios

### 2. Feature Engineering
- **Provider-Level Aggregation:**
  - Claims per patient ratio
  - Inpatient/Outpatient ratios
  - Average reimbursement metrics
  - Chronic condition counts
  - Weekend and same-day claim patterns

- **Feature Selection:**
  - Correlation analysis (removed features with >0.9 correlation)
  - Random Forest feature importance
  - Final selection: Top 25 most important features

### 3. Handling Class Imbalance
- Class weighting (`class_weight='balanced'`)
- Focused on Precision, Recall, F1-score, and ROC-AUC metrics

### 4. Model Training

Three models were evaluated:

#### Logistic Regression
- **Accuracy:** 86%
- **ROC-AUC:** 0.92
- **Precision (Fraud):** 45%
- **Recall (Fraud):** 86%
- **F1-Score (Fraud):** 59%

#### Decision Tree
- **Accuracy:** 90%
- **ROC-AUC:** 0.79
- **Precision (Fraud):** 57%
- **Recall (Fraud):** 65%
- **F1-Score (Fraud):** 61%

#### Gradient Boosting (BEST MODEL) 🏆
- **Accuracy:** 93%
- **ROC-AUC:** 0.93
- **Precision (Fraud):** 72%
- **Recall (Fraud):** 62%
- **F1-Score (Fraud):** 67%

##  Results

### Model Comparison Summary

| Metric | Logistic Regression | Decision Tree | Gradient Boosting |
|--------|---------------------|---------------|-------------------|
| Accuracy | 86% | 90% | **93%** |
| ROC-AUC | 0.92 | 0.79 | **0.93** |
| Precision (Fraud) | 45% | 57% | **72%** |
| Recall (Fraud) | 86% | 65% | 62% |
| F1-Score (Fraud) | 59% | 61% | **67%** |

### Key Findings:
- **Winner:** Gradient Boosting achieves the best overall performance
- **72% precision** means only 28% false positive rate (vs 55% for Logistic Regression)
- **Highest F1-scores** for both classes indicate balanced performance
- Trade-off: Slightly lower recall (62%) compared to Logistic Regression (86%)

##  Usage

### Running the Notebooks

1. **Data Engineering:**
```bash
jupyter notebook notebooks/data_engineering.ipynb
```
This notebook performs data cleaning, feature engineering, and prepares the final dataset.

2. **Modeling:**
```bash
jupyter notebook notebooks/Modeling.ipynb
```
This notebook trains all three models and evaluates their performance.

### Making Predictions

```python
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer

# Load your trained model
model = GradientBoostingClassifier(random_state=42)
# ... (train model as shown in notebook)

# Make predictions
new_data = pd.read_csv('new_provider_data.csv')
imputer = SimpleImputer(strategy='mean')
new_data_imputed = imputer.fit_transform(new_data)
predictions = model.predict(new_data_imputed)
fraud_probabilities = model.predict_proba(new_data_imputed)[:, 1]
```

##  Key Features

- **Comprehensive Data Cleaning:** Handles multiple data sources with thorough validation
- **Advanced Feature Engineering:** Creates 55+ provider-level features
- **Feature Selection:** Reduces dimensionality while maintaining predictive power
- **Class Imbalance Handling:** Uses class weighting to address 1:7.4 imbalance
- **Model Comparison:** Evaluates multiple algorithms systematically
- **Interpretable Results:** Feature importance analysis included

##  Future Improvements

1. **Hyperparameter Tuning:** Use GridSearchCV/RandomizedSearchCV for optimal parameters
2. **Ensemble Methods:** Combine multiple models for better performance
3. **Error Analysis:** Deep dive into false positives and false negatives
4. **Threshold Optimization:** Adjust decision threshold based on business costs
5. **External Data:** Incorporate additional data sources for better predictions


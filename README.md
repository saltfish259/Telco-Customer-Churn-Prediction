# 📡 Telco Customer Churn Prediction

A complete end-to-end machine learning project that predicts whether a telecom customer is likely to churn. Built with LightGBM and deployed as an interactive Streamlit application.

---

## 🚀 Live Demo

[**Try the Streamlit App →**](https://telco-customer-churn-prediction-project1.streamlit.app/)

---

## Overview

Customer churn occurs when a customer stops using a service. Predicting churn early gives a business the opportunity to act before losing the customer.

This project uses the **Telco Customer Churn** dataset to build a binary classification model. After comparing several classification models, **LightGBM** was selected as the best performer. The final model is wrapped in a preprocessing pipeline and deployed through a Streamlit web application.

---

## Project Goals

1. Explore the dataset and understand its structure.
2. Clean and prepare the data for modeling.
3. Benchmark multiple classification models.
4. Identify the best model for this dataset.
5. Discover suitable LightGBM parameters through a structured search.
6. Build a final preprocessing + model pipeline.
7. Save the pipeline for deployment.
8. Deploy the model through a Streamlit application.

---

## Workflow

```
EDA
 ↓
Data Cleaning
 ↓
Baseline Benchmark
 ↓
LightGBM Parameter Discovery
 ↓
Final Model + Pipeline
 ↓
Streamlit Application
```

| Stage | Purpose |
|---|---|
| **EDA** | Understand the dataset before making any changes |
| **Data Cleaning** | Fix data quality issues and prepare a clean dataset |
| **Baseline Benchmark** | Compare multiple models and select the best one |
| **Parameter Discovery** | Search for a better LightGBM configuration |
| **Final Model + Pipeline** | Train the final model and save a deployment-ready pipeline |
| **Streamlit Application** | Allow users to interactively test churn predictions |

---

## Dataset

| Item | Detail |
|---|---|
| Dataset | Telco Customer Churn |
| Original File | `WA_Fn-UseC_-Telco-Customer-Churn.csv` |
| Cleaned File | `clean_dataset.csv` |

The cleaned dataset is the version used for all model development steps.

---

## Project Structure

```
project-root/
│
├── notebooks/
│   ├── EDA_Classification_Notebook.ipynb
│   ├── Cleaning_Classification_Notebook.ipynb
│   ├── Baseline_Benchmark_Notebook.ipynb
│   ├── LightGBM_Parameter_Discovery_Notebook.ipynb
│   └── Final_Model_and_Deployment_Pipeline.ipynb
│
├── data/
│   └── clean_dataset.csv
│
├── models/
│   └── lightgbm_pipeline.pkl
│
├── app.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

---

## Notebooks

### 1. EDA Classification Notebook

Explores the raw dataset before any cleaning or modeling.

- Inspect dataset structure and data types
- Identify missing values and duplicate rows
- Detect potential outliers
- Understand target distribution
- Explore feature relationships

### 2. Cleaning Classification Notebook

Prepares a clean, model-ready dataset.

- Remove unnecessary columns
- Handle missing values
- Fix incorrect data types
- Save the cleaned dataset

### 3. Baseline Benchmark Notebook

Tests and compares multiple classification models on the same data.

- Apply label encoding and one-hot encoding
- Split data into training and test sets
- Apply SMOTE to the training data to handle class imbalance
- Train and compare: Logistic Regression, Decision Tree, Random Forest, Extra Trees, XGBoost, LightGBM, and CatBoost

> **Result:** LightGBM performed best and was selected for further development.

### 4. LightGBM Parameter Discovery Notebook

Searches for a better LightGBM configuration using RandomizedSearchCV.

- Define a parameter search space
- Test 50 candidate combinations with 5-fold cross-validation
- Evaluate each candidate using ROC AUC, F1, Recall, and Precision
- Compare the top candidates side by side
- Select a balanced parameter configuration

### 5. Final Model and Deployment Pipeline Notebook

Prepares the final model for deployment.

- Recreate the required preprocessing steps
- Train the final LightGBM model using the selected parameters
- Evaluate the final model on the test set
- Combine preprocessing and model into a single Scikit-learn Pipeline
- Save the pipeline as `lightgbm_pipeline.pkl`

---

## Features

The final model uses these 10 features:

| Feature | Type |
|---|---|
| InternetService | Categorical |
| PaymentMethod | Categorical |
| Contract | Categorical |
| MultipleLines | Categorical |
| StreamingTV | Categorical |
| StreamingMovies | Categorical |
| PhoneService | Categorical |
| tenure | Numeric |
| MonthlyCharges | Numeric |
| TotalCharges | Numeric |

- Categorical features are **encoded automatically** by the preprocessing pipeline.
- **Feature scaling is not applied** — LightGBM is a tree-based model and does not require it.
- **SMOTE is used only during training.** It is not applied during Streamlit prediction.

---

## Model

The final model is **LightGBM** — selected because it produced the best results during the baseline model comparison for this dataset.

### Final Parameters

| Parameter | Value | Purpose |
|---|---|---|
| learning_rate | 0.01 | Controls how fast the model learns |
| n_estimators | 500 | Number of boosting trees |
| num_leaves | 15 | Controls tree complexity |
| max_depth | -1 | No depth limit |
| min_child_samples | 10 | Minimum samples required in a leaf |
| subsample | 0.8 | Fraction of rows used per tree |
| colsample_bytree | 0.7 | Fraction of features used per tree |
| reg_alpha | 0 | L1 regularization |
| reg_lambda | 0 | L2 regularization |

---

## Model Performance

### Baseline LightGBM (default parameters)

| Metric | Score |
|---|---|
| ROC AUC | 0.8279 |
| F1 Score | 0.7850 |
| Precision | 0.7874 |
| Recall | 0.7829 |
| Accuracy | 0.7829 |

---

## Parameter Discovery

RandomizedSearchCV tested 50 different parameter combinations. The highest ROC AUC found was **0.8362**.

### Recommended Candidate

| Metric | Score |
|---|---|
| ROC AUC | 0.8360 |
| F1 Score | 0.6274 |
| Recall | 0.7016 |
| Precision | 0.5674 |

### Why wasn't the highest ROC AUC candidate selected?

The candidate with the top ROC AUC of 0.8362 was not automatically chosen. A slightly lower-ranked candidate was selected instead because it produced a **better balance across all four metrics** — particularly a higher Recall.

In churn prediction, failing to identify an actual churner is more costly than a false alarm. A model with marginally lower ROC AUC but stronger Recall can be more useful in practice.

> This demonstrates that model selection should consider the full picture, not just a single metric.

---

## Final Pipeline

The saved file `lightgbm_pipeline.pkl` contains two steps bundled into one object:

1. **Preprocessing** — handles categorical encoding automatically.
2. **Trained LightGBM model** — produces the churn prediction.

When the Streamlit application receives a user's input, it passes the raw values directly into the pipeline. The pipeline handles all required transformations before making a prediction — no manual preprocessing is needed in the application code.

This approach makes deployment simpler, cleaner, and less error-prone.

---

## Streamlit Application

The application lets users test the trained model interactively.

**User inputs:**
- Select categorical customer information using dropdowns.
- Enter numeric values for tenure and monthly charges.
- `TotalCharges` is **calculated automatically** from `tenure × MonthlyCharges`.

**Application output:**

| Output | Description |
|---|---|
| Predicted Class | Churn or Not Churn |
| Churn Probability | Model confidence as a percentage |
| Risk Level | Simple interpretation of the probability |

**Risk level thresholds:**

| Probability | Risk Level |
|---|---|
| Below 30% | 🟢 Low |
| 30% to below 60% | 🟡 Medium |
| 60% or higher | 🔴 High |

### Example Test Results

**High-risk customer:**
```
Prediction:        Churn
Churn Probability: 78.5%
Risk Level:        High
```

**Low-risk customer:**
```
Prediction:        Not Churn
Churn Probability: 2.7%
Risk Level:        Low
```

> These are example results used to verify the application. They are not guaranteed predictions for real customers.

---

## What This Project Demonstrates

| Skill | Applied |
|---|---|
| End-to-end ML workflow | EDA → Cleaning → Modeling → Deployment |
| Classification modeling | Binary churn classification |
| Feature preprocessing | Encoding, type handling |
| Handling categorical data | Label encoding + one-hot encoding |
| Handling class imbalance | SMOTE on training data |
| Model benchmarking | 7 models compared |
| Parameter discovery | RandomizedSearchCV with 50 candidates |
| Multi-metric model evaluation | ROC AUC, F1, Recall, Precision |
| Pipeline design | Scikit-learn ColumnTransformer + Pipeline |
| Model serialization | Joblib `.pkl` export |
| Streamlit deployment | Interactive prediction web app |

---

## Limitations

- This is an **educational portfolio project**, not a production system.
- The model was trained on one specific dataset and may not generalize to other telecom companies or customer populations.
- Churn probability is a model-based estimate, not a certainty.
- Business decisions should not rely solely on this model's output.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

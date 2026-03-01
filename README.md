# Cognify — Alzheimer's Early Risk Assessment

> An end-to-end machine learning web application that predicts Alzheimer's disease risk from clinical and behavioral inputs, with explainable AI and downloadable medical reports.

🔗 **Live Demo:** [cognify1.streamlit.app](https://cognify1.streamlit.app)
📁 **Dataset:** [Alzheimer's Disease Dataset — Kaggle](https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset)

---

## What is Cognify?

Cognify is a clinical decision-support tool built to assist in early screening of Alzheimer's disease. A user (patient or caregiver) enters 10 key clinical and behavioral indicators, and the model returns a risk prediction, confidence score, and a detailed explanation of which factors contributed most to the result — all downloadable as a structured PDF report.

---

## Features

- **Risk Prediction** — Binary classification (Alzheimer's / No Alzheimer's) with probability score
- **Risk Meter** — Visual gauge showing Low / Moderate / High risk level
- **SHAP Explainability** — Feature contribution chart showing why the model made its decision
- **PDF Report** — Downloadable lab-style report with patient inputs, diagnosis, and risk breakdown
- **Clean UI** — Minimal, professional interface built with Streamlit

---

## Machine Learning Pipeline

### 1. Exploratory Data Analysis
- Class distribution analysis (65/35 imbalance detected)
- Correlation heatmap — identified MMSE, ADL, FunctionalAssessment as strong predictors
- Boxplots, bar charts, pairplots for feature understanding
- Outlier analysis on continuous features

### 2. Preprocessing
- Dropped non-predictive identifiers (PatientID, DoctorInCharge)
- Stratified train/test split (80/20) to preserve class ratio
- StandardScaler applied — fit on training data only to prevent data leakage

### 3. Class Imbalance
- Used `class_weight='balanced'` inside models instead of SMOTE
- Appropriate for mild imbalance (65/35) — no synthetic data needed

### 4. Model Training & Comparison

| Model | Accuracy | Recall (AD) | Precision (AD) | F1 (AD) |
|-------|----------|-------------|----------------|---------|
| Logistic Regression (baseline) | 82% | 86% | 70% | 77% |
| Random Forest | 93% | 85% | 94% | 89% |
| XGBoost (untuned) | 82% | 86% | 70% | 77% |
| **XGBoost (tuned)** | **95%** | **94%** | **91%** | **93%** |

> Recall was prioritized as the key metric — missing an Alzheimer's patient (false negative) is more dangerous than a false alarm.

### 5. Hyperparameter Tuning
- GridSearchCV with 5-fold cross validation on XGBoost
- Best parameters: `learning_rate=0.1`, `max_depth=9`, `n_estimators=100`

### 6. Feature Selection
- Trained final model on top 10 features by importance score
- Reduced from 33 → 10 features with accuracy improvement (95% vs 94%)
- Proves remaining 23 features were adding noise, not signal

### 7. Feature Importance (Top 10)

| Feature | Importance |
|---------|------------|
| Memory Complaints | 0.20 |
| Behavioral Problems | 0.14 |
| Functional Assessment | 0.12 |
| MMSE Score | 0.11 |
| ADL Score | 0.10 |
| Forgetfulness | 0.02 |
| Smoking | 0.017 |
| Age | 0.016 |
| Cardiovascular Disease | 0.016 |
| BMI | 0.014 |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Language | Python 3.x |
| ML | scikit-learn, XGBoost |
| Explainability | SHAP |
| Visualization | Matplotlib, Plotly |
| App | Streamlit |
| PDF Generation | ReportLab |
| Model Persistence | Joblib |
| Version Control | Git + GitHub |
| Deployment | Streamlit Cloud |

---

## Project Structure

```
cognify/
├── app.py                          # Streamlit application
├── model_training.ipynb            # Full ML pipeline notebook
├── eda.ipynb                       # Exploratory data analysis notebook
├── alzheimer_model.pkl             # Trained XGBoost model
├── scaler.pkl                      # Fitted StandardScaler
├── requirements.txt                # Python dependencies
├── dataset/
│   └── alzheimers_disease_data.csv
└── README.md
```

---

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/momojaffer1804/Cognify.git
cd Cognify

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

App will open at `http://localhost:8501`

---

## Key Learnings

- Why recall matters more than accuracy in medical diagnosis
- How data leakage occurs and how to prevent it (fit scaler on train only)
- Why untuned XGBoost can perform same as Logistic Regression
- Feature selection can improve accuracy by removing noise features
- SHAP values for clinical explainability of black-box models

---

## Disclaimer

Cognify is an AI-assisted screening tool built for educational purposes. It does not constitute a medical diagnosis. Always consult a qualified neurologist for clinical assessment.

---

## Author

**Rehan** — Built as an end-to-end ML portfolio project

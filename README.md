# CodeAlpha Credit Scoring Model

## Objective
Predict a customer's creditworthiness using financial/credit-history data as a binary classification problem.

## Internship Task
CodeAlpha Machine Learning Internship — Task 1: Credit Scoring Model.

## Dataset
This project uses the **UCI Statlog (German Credit Data)** dataset. The dataset contains credit-related attributes such as checking account status, credit duration, credit history, credit amount, savings, employment, age, housing, existing credits, and other customer information.

The notebook downloads the public dataset from the UCI repository and assigns meaningful column names.

## Workflow
1. Load dataset
2. Data inspection
3. Data cleaning
4. Exploratory Data Analysis (EDA)
5. Categorical encoding
6. Train/test split
7. Feature scaling
8. Train Logistic Regression, Decision Tree and Random Forest
9. Compare models
10. Evaluate using Accuracy, Precision, Recall, F1-score and ROC-AUC
11. Save the best model
12. Run a Streamlit prediction app

## Run
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Open:
`notebooks/credit_scoring_model.ipynb`

After training, run:
```bash
streamlit run app/app.py
```

## Repository Name
`CodeAlpha_CreditScoringModel`

## Important
This is an educational machine-learning project, not a real lending or financial decision system.

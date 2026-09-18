import pandas as pd
from pathlib import Path

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"

COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings_status", "employment", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since",
    "property", "age", "other_installment_plans", "housing",
    "existing_credits", "job", "dependents", "telephone", "foreign_worker",
    "target"
]

def main():
    out = Path(__file__).resolve().parents[1] / "data" / "german_credit.csv"
    df = pd.read_csv(URL, sep=r"\s+", header=None, names=COLUMNS)
    # UCI target: 1 = good credit, 2 = bad credit
    df["target"] = df["target"].map({1: 1, 2: 0})
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")

if __name__ == "__main__":
    main()

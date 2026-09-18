import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve
)

# -----------------------------
# 1. Load Dataset
# -----------------------------
DATA_PATH = "data/german_credit.csv"

df = pd.read_csv(DATA_PATH)

X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------
# 2. Identify Columns
# -----------------------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

# -----------------------------
# 3. Preprocessing
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 5. Models
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
}

results = []
trained_models = {}

# -----------------------------
# 6. Train & Evaluate
# -----------------------------
for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc
    })

    trained_models[name] = pipeline


# -----------------------------
# 7. Results
# -----------------------------
results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

# -----------------------------
# 8. Select Best Model
# -----------------------------
best_model_name = results_df.loc[
    results_df["F1"].idxmax(),
    "Model"
]

best_model = trained_models[best_model_name]

print(f"\nBest model: {best_model_name}")

# -----------------------------
# 9. Save Best Model
# -----------------------------
os.makedirs("models", exist_ok=True)

model_path = "models/credit_scoring_model.joblib"

joblib.dump(best_model, model_path)

print(f"Saved to: {os.path.abspath(model_path)}")

# -----------------------------
# 10. Confusion Matrix
# -----------------------------
best_predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Bad Credit", "Good Credit"]
)

disp.plot()

plt.title(f"Confusion Matrix - {best_model_name}")
plt.tight_layout()

os.makedirs("screenshots", exist_ok=True)

plt.savefig(
    "screenshots/confusion_matrix.png",
    dpi=300
)

plt.show()

# -----------------------------
# 11. ROC Curve
# -----------------------------
best_probabilities = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test,
    best_probabilities
)

plt.figure(figsize=(7, 5))

plt.plot(
    fpr,
    tpr,
    label=f"{best_model_name} (AUC = {roc_auc_score(y_test, best_probabilities):.3f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()
plt.tight_layout()

plt.savefig(
    "screenshots/roc_curve.png",
    dpi=300
)

plt.show()

print("\nEvaluation graphs saved successfully!")
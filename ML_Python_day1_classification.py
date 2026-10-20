#!/usr/bin/env python3
"""Day 1: breast cancer classification. Irina Chelysheva, October 2026.

Run the whole script, or explore its numbered sections in Spyder/Jupyter.
Keep this file and breast_cancer_wisconsin_original.csv in the same folder.
Historical teaching data, not a clinically validated diagnostic system.
"""

# %% 1. Libraries and settings
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import (GroupShuffleSplit, StratifiedGroupKFold, cross_validate)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = BASE_DIR / "outputs_classification"
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 1
N_SPLITS = 5
K_FEATURES = "all"  # Exercise: try 5. Selection is fitted inside each CV fold.


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")


# %% 2. Load and explore: one row per observation, nine predictors
df = pd.read_csv(BASE_DIR / "breast_cancer_wisconsin_original.csv")
X = df.drop(columns=["Id", "Class"])
y = df["Class"]  # This CSV uses 0 = benign, 1 = malignant.
groups = df["Id"]  # Repeated sample codes must stay in the same partition.
print(f"Observations: {len(df)}; predictors: {X.shape[1]}; unique IDs: {groups.nunique()}")
print(df.head())
print(X.describe())
print("\nClass counts (0 = benign, 1 = malignant):\n", y.value_counts().sort_index())
print("\nMissing predictor values:\n", X.isna().sum())
assert set(y.unique()) == {0, 1}, "Check the class encoding in the CSV."

# %% 3. Hold out approximately 30% of sample IDs for final testing
# GroupShuffleSplit preserves IDs, but does not guarantee exact class proportions.
splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=RANDOM_STATE)
train_idx, test_idx = next(splitter.split(X, y, groups))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
groups_train, groups_test = groups.iloc[train_idx], groups.iloc[test_idx]
assert set(groups_train).isdisjoint(set(groups_test))
print(f"\nTrain observations: {len(X_train)}; test observations: {len(X_test)}")
print("Train class proportions:\n", y_train.value_counts(normalize=True).sort_index())
print("Test class proportions:\n", y_test.value_counts(normalize=True).sort_index())

# %% 4. A complete pipeline: impute, scale, optionally select, then classify
# Every fit occurs on the training partition of each CV fold.
# Scaling matters for LR, KNN and SVM. It is harmless for ordinary trees.
def make_model_pipeline(estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selection", SelectKBest(score_func=f_classif, k=K_FEATURES)),
        ("model", estimator),
    ])

models = {
    "Baseline": DummyClassifier(strategy="prior"),
    "LR": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "KNN": KNeighborsClassifier(),
    "DTC": DecisionTreeClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(kernel="rbf", gamma="scale", probability=True, random_state=RANDOM_STATE),
    "RF": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
}
pipelines = {name: make_model_pipeline(model) for name, model in models.items()}

# %% 5. Compare models using training data only
# Stratification approximately preserves class proportions; grouping preserves IDs.
cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=7)
scoring = {"accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy", "roc_auc": "roc_auc"}
rows, fold_auc = [], {}
for name, pipeline in pipelines.items():
    scores = cross_validate(
        pipeline, X_train, y_train, groups=groups_train, cv=cv,
        scoring=scoring, error_score="raise", n_jobs=1,
    )
    row = {"model": name}
    for metric in scoring:
        row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
        row[f"{metric}_sd"] = scores[f"test_{metric}"].std(ddof=1)
    rows.append(row)
    fold_auc[name] = scores["test_roc_auc"]

cv_results = pd.DataFrame(rows).set_index("model").sort_values("roc_auc_mean", ascending=False)
print("\nTraining cross-validation results (SD describes variation across folds):")
print(cv_results.round(3).to_string())
cv_results.to_csv(OUTPUT_DIR / "cv_results.csv")

plt.figure(figsize=(8, 5))
plt.boxplot(list(fold_auc.values()), tick_labels=list(fold_auc))
plt.ylabel("Validation ROC AUC")
plt.title("Five-fold cross-validation on training data")
save_figure("model_comparison.png")

# %% 6. Select by mean CV ROC AUC, then refit on all training observations
best_name = cv_results.index[0]
best_model = clone(pipelines[best_name]).fit(X_train, y_train)
selected_features = X.columns[best_model.named_steps["selection"].get_support()]
print(f"\nSelected model: {best_name}")
print("Selected features:", list(selected_features))

# %% 7. Final held-out evaluation: malignant is the positive class
predictions = best_model.predict(X_test)
positive_index = list(best_model.classes_).index(1)
probabilities = best_model.predict_proba(X_test)[:, positive_index]
cm = confusion_matrix(y_test, predictions, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()
metrics = {
    "model": best_name,
    "accuracy": float(accuracy_score(y_test, predictions)),
    "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
    "sensitivity": float(tp / (tp + fn)),
    "specificity": float(tn / (tn + fp)),
    "roc_auc": float(roc_auc_score(y_test, probabilities)),
}
print("\nFinal test metrics:\n", json.dumps(metrics, indent=2))
print(classification_report(y_test, predictions, labels=[0, 1], target_names=["Benign", "Malignant"], zero_division=0))
print("\nDiscussion: which error matters more here, a false negative or a false positive?")

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
plt.xlabel("Predicted class")
plt.ylabel("Actual class")
plt.title(f"Held-out confusion matrix: {best_name}")
save_figure("confusion_matrix.png")

fpr, tpr, _ = roc_curve(y_test, probabilities, pos_label=1)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"{best_name}: AUC = {metrics['roc_auc']:.3f}")
plt.plot([0, 1], [0, 1], "k--", label="Chance")
plt.xlabel("False positive rate (1 − specificity)")
plt.ylabel("True positive rate (sensitivity)")
plt.title("Held-out ROC curve")
plt.legend()
save_figure("roc_curve.png")

# %% 8. Interpret after final evaluation; do not tune against this test set
# Permute each original predictor and measure the reduction in test ROC AUC.
# This is predictive importance, not evidence of causality. Correlated predictors
# may share importance. Negative values can occur through sampling variation.
importance = permutation_importance(
    best_model, X_test, y_test, scoring="roc_auc", n_repeats=10,
    random_state=RANDOM_STATE, n_jobs=1,
)
importance_df = pd.DataFrame({
    "feature": X.columns, "mean_auc_decrease": importance.importances_mean,
    "sd_auc_decrease": importance.importances_std,
}).sort_values("mean_auc_decrease")
print("\nPermutation importance:\n", importance_df.to_string(index=False))
plt.figure(figsize=(8, 5))
plt.barh(importance_df["feature"], importance_df["mean_auc_decrease"],
         xerr=importance_df["sd_auc_decrease"], color="steelblue")
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Mean decrease in test ROC AUC after permutation")
plt.title("Predictive feature importance")
save_figure("feature_importance.png")

# %% 9. Save the WHOLE fitted pipeline for consistent future preprocessing
joblib.dump(best_model, OUTPUT_DIR / "best_classification_pipeline.joblib")
pd.DataFrame({"row_index": X_test.index, "sample_id": groups_test,
              "actual": y_test, "predicted": predictions,
              "probability_malignant": probabilities}).to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)
importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
(OUTPUT_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"\nFigures, results and fitted pipeline saved to {OUTPUT_DIR}")
plt.show()

# Exercises: use training CV to compare K_FEATURES='all' with K_FEATURES=5;
# try different KNN neighbour counts or tree depths. Once you have inspected the
# test results, further experiments on this same split are exploratory, rather
# than a new independent test of generalisation.

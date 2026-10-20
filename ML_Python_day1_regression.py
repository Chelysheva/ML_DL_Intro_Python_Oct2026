#!/usr/bin/env python3
"""Day 1: diabetes progression regression. Irina Chelysheva, October 2026.

Keep this script and diabetes_regression.csv in the same folder.
442 patients, 10 baseline predictors, continuous progression score at one year.
This score is not HbA1c, a probability, or a percentage. Historical teaching data.
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
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
OUTPUT_DIR = BASE_DIR / "outputs_regression"
OUTPUT_DIR.mkdir(exist_ok=True)
RANDOM_STATE = 1
N_SPLITS = 5
K_FEATURES = "all"  # Exercise: try 5; selection stays inside CV.


def save_figure(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=160, bbox_inches="tight")


# %% 2. Load data and separate the predictors from the target
df = pd.read_csv(BASE_DIR / "diabetes_regression.csv")
X = df.drop(columns="disease_progression")
y = df["disease_progression"]
print(f"Patients: {len(df)}; predictors: {X.shape[1]}")
print(df.head())
print(X.describe())
print("\nMissing values:\n", df.isna().sum())

# %% 3. Set aside the test data before exploration or fitted preprocessing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE)
print(f"\nTraining patients: {len(X_train)}; test patients: {len(X_test)}")
train_data = X_train.assign(disease_progression=y_train)
correlations = train_data.corr()
plt.figure(figsize=(11, 9))
sns.heatmap(correlations, annot=True, fmt=".2f", cmap="vlag", center=0,
            vmin=-1, vmax=1, annot_kws={"size": 8})
plt.title("Training-data correlations")
save_figure("training_correlations.png")
print("\nTraining correlations with progression:\n",
      correlations["disease_progression"].drop("disease_progression").sort_values())
print("\nCorrelation shows association, not causation.")

# %% 4. Impute, standardise, optionally select, then fit
# The CSV contains source-scale predictors; s5 is a source-supplied
# triglyceride-related measure, usually described as log serum triglycerides.
# StandardScaler learns its mean and SD separately within each training fold.
def make_model_pipeline(estimator):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("selection", SelectKBest(score_func=f_regression, k=K_FEATURES)),
        ("model", estimator),
    ])

models = {
    "Baseline": DummyRegressor(strategy="mean"),
    "LR": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=1.0, max_iter=10000),
    "ElasticNet": ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000),
    "KNN": KNeighborsRegressor(),
    "DTR": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "SVR": SVR(kernel="rbf", C=100.0, epsilon=1.0),
    "RF": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1),
}
# These are illustrative settings, not an exhaustive hyperparameter search.
pipelines = {name: make_model_pipeline(model) for name, model in models.items()}

# %% 5. Compare on five identical validation folds within the training set
cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=7)
scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
rows, fold_rmse = [], {}
for name, pipeline in pipelines.items():
    scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring,
                            error_score="raise", n_jobs=1)
    # sklearn negates losses so that larger scores are better. Convert back to
    # ordinary positive errors for reporting: lower RMSE/MAE is better.
    rmse, mae = -scores["test_rmse"], -scores["test_mae"]
    rows.append({"model": name, "rmse_mean": rmse.mean(), "rmse_sd": rmse.std(ddof=1),
                 "mae_mean": mae.mean(), "r2_mean": scores["test_r2"].mean()})
    fold_rmse[name] = rmse

cv_results = pd.DataFrame(rows).set_index("model").sort_values("rmse_mean")
print("\nTraining cross-validation results (lower RMSE is better):")
print(cv_results.round(3).to_string())
cv_results.to_csv(OUTPUT_DIR / "cv_results.csv")
plt.figure(figsize=(10, 5))
plt.boxplot(list(fold_rmse.values()), tick_labels=list(fold_rmse))
plt.ylabel("Validation RMSE (progression-score units)")
plt.title("Five-fold cross-validation on training data")
save_figure("model_comparison.png")

# %% 6. Choose by mean CV RMSE, then fit on all training patients
best_name = cv_results.index[0]
best_model = clone(pipelines[best_name]).fit(X_train, y_train)
print(f"\nSelected model: {best_name}")
print("Selected features:", list(X.columns[best_model.named_steps["selection"].get_support()]))

# %% 7. Evaluate once on the held-out patients, in the original target units
predictions = best_model.predict(X_test)
metrics = {
    "model": best_name,
    "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
    "mae": float(mean_absolute_error(y_test, predictions)),
    "r2": float(r2_score(y_test, predictions)),
}
baseline = DummyRegressor(strategy="mean").fit(X_train, y_train)
baseline_predictions = baseline.predict(X_test)
metrics["baseline_rmse"] = float(np.sqrt(mean_squared_error(y_test, baseline_predictions)))
print("\nFinal test metrics:\n", json.dumps(metrics, indent=2))
print("RMSE and MAE are in progression-score units; R² can be negative.")

plt.figure(figsize=(6, 6))
plt.scatter(y_test, predictions, alpha=0.7)
limits = [min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())]
plt.plot(limits, limits, "k--", label="Perfect prediction")
plt.xlabel("Observed progression score")
plt.ylabel("Predicted progression score")
plt.title(f"Held-out predictions: {best_name}")
plt.legend()
save_figure("actual_vs_predicted.png")

residuals = y_test.to_numpy() - predictions
plt.figure(figsize=(7, 5))
plt.scatter(predictions, residuals, alpha=0.7)
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Predicted progression score")
plt.ylabel("Residual (observed − predicted)")
plt.title("Held-out residuals")
save_figure("residuals.png")

# %% 8. Inspect predictive importance after final evaluation
# Do not use these test-data results to select new features or tune the model.
# Correlated predictors can share importance; importance is not a causal effect.
importance = permutation_importance(
    best_model, X_test, y_test, scoring="neg_root_mean_squared_error",
    n_repeats=10, random_state=RANDOM_STATE, n_jobs=1,
)
importance_df = pd.DataFrame({"feature": X.columns,
    "mean_rmse_increase": importance.importances_mean,
    "sd_rmse_increase": importance.importances_std}).sort_values("mean_rmse_increase")
print("\nPermutation importance:\n", importance_df.to_string(index=False))
plt.figure(figsize=(8, 5))
plt.barh(importance_df["feature"], importance_df["mean_rmse_increase"],
         xerr=importance_df["sd_rmse_increase"], color="forestgreen")
plt.axvline(0, color="black", linewidth=0.8)
plt.xlabel("Mean increase in test RMSE after permutation")
plt.title("Predictive feature importance")
save_figure("feature_importance.png")

# %% 9. Save the full pipeline, numerical results and figures
joblib.dump(best_model, OUTPUT_DIR / "best_regression_pipeline.joblib")
pd.DataFrame({"row_index": X_test.index, "actual": y_test,
              "predicted": predictions, "residual": residuals}).to_csv(
                  OUTPUT_DIR / "test_predictions.csv", index=False)
importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)
(OUTPUT_DIR / "test_metrics.json").write_text(json.dumps(metrics, indent=2))
print(f"\nFigures, results and fitted pipeline saved to {OUTPUT_DIR}")
plt.show()

# Exercises: compare K_FEATURES='all' with 5 using training CV; change Ridge's
# alpha; compare training and validation performance as tree depth increases.
# Repeated experiments after inspecting the test set are exploratory.

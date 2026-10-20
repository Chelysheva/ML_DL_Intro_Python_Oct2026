# Day 1 datasets

Both CSV files can be loaded offline. Neither contains a saved pandas row-index column. These historical datasets support teaching and algorithm comparisons; the resulting models are not validated clinical tools.

## Diabetes progression: `diabetes_regression.csv`

442 patients, 10 baseline predictors and one continuous target. Exported from the dataset bundled with scikit-learn 1.8.0 using:

```python
from sklearn.datasets import load_diabetes
dataset = load_diabetes(as_frame=True, scaled=False)
dataset.frame.rename(columns={"target": "disease_progression"}).to_csv(
    "diabetes_regression.csv", index=False
)
```

`scaled=False` returns predictors on the source scale rather than the default full-dataset standardisation. The script fits its own standardisation separately within each training fold. Source-scale does not mean that every variable is an untransformed laboratory measurement.

| Column | Meaning |
| --- | --- |
| `age` | Age in years |
| `sex` | Numeric sex code, retained as supplied; no coding interpretation is assumed in this exercise |
| `bmi` | Body mass index |
| `bp` | Average blood pressure |
| `s1` | Total serum cholesterol (`tc`) |
| `s2` | Low-density lipoproteins (`ldl`) |
| `s3` | High-density lipoproteins (`hdl`) |
| `s4` | Total cholesterol / HDL ratio (`tch`) |
| `s5` | Triglyceride-related measure (`ltg`), usually described as log serum triglycerides; see the documentation caveat below |
| `s6` | Blood sugar (`glu`) |
| `disease_progression` | Supplied quantitative measure of disease progression one year after baseline |

The target ranges from 25 to 346 in this copy. It is not HbA1c, a probability, a percentage, a diagnosis, or a prediction of whether diabetes will develop. All observations are from people with diabetes. Do not infer clinical thresholds or target units beyond those provided by the original documentation.

Scikit-learn explicitly cautions that the original source does not fully explain all feature meanings, especially `ltg`. Preserve those limitations when discussing the measurements. The CSV has no missing values. Median imputation remains in the pipeline to demonstrate a reusable workflow.

Sources and attribution:

- [scikit-learn dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html)
- [Original dataset page, hosted by NC State](https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html)
- Efron, B., Hastie, T., Johnstone, I. and Tibshirani, R. (2004). Least Angle Regression. *The Annals of Statistics*, 32(2), pp. 407–499. [doi:10.1214/009053604000000067](https://doi.org/10.1214/009053604000000067).
- [scikit-learn licence](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) (BSD-3-Clause). Retain dataset attribution and consult the original source for dataset-specific information.

## Breast cancer: `breast_cancer_wisconsin_original.csv`

This is the same data source used by the previous classification script, now included locally to avoid a live download during class.

- 699 observations and nine predictive features.
- 645 distinct `Id` values. Some IDs repeat; the script keeps each ID together for the train/test split and cross-validation.
- 458 benign observations and 241 malignant observations.
- 16 missing values, all in `Bare.nuclei`.
- This is Wisconsin Breast Cancer **Original**, not the separate Wisconsin Diagnostic dataset.

| Column | Meaning |
| --- | --- |
| `Id` | Sample code, used only for grouped splitting, never as a predictor |
| `Cl.thickness` | Clump thickness |
| `Cell.size` | Uniformity of cell size |
| `Cell.shape` | Uniformity of cell shape |
| `Marg.adhesion` | Marginal adhesion |
| `Epith.c.size` | Single epithelial cell size |
| `Bare.nuclei` | Bare nuclei |
| `Bl.cromatin` | Bland chromatin (column spelling retained from the mirror) |
| `Normal.nucleoli` | Normal nucleoli |
| `Mitoses` | Mitoses |
| `Class` | **0 = benign, 1 = malignant** in this CSV |

The nine predictors use integer scores from 1 to 10, apart from missing values. The mirror recodes the original UCI class labels (2 = benign, 4 = malignant) as 0/1. This copy retains all rows and the mirror's column names and values. CSV serialisation represents missing values as empty fields.

Sources and attribution:

- William Wolberg, University of Wisconsin Hospitals. [UCI Breast Cancer Wisconsin (Original)](https://archive.ics.uci.edu/dataset/15/breast+cancer+wisconsin+original), [doi:10.24432/C5HP4Z](https://doi.org/10.24432/C5HP4Z).
- UCI lists the dataset under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Attribution is retained here; changes are described above.
- [CSV mirror used in the previous course script](https://raw.githubusercontent.com/selva86/datasets/master/BreastCancer.csv).

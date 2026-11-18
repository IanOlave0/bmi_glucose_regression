# Pima Diabetes — Regression and Classification from Scratch

Learning record for the Machine Learning Specialization (MLS C1): regression and regularized logistic regression implemented from scratch with NumPy, no `sklearn`.

Dataset: Pima Indians Diabetes — https://www.kaggle.com/datasets/mragpavank/diabetes.
Zeros in `BMI`, `Glucose`, `Age`, `BloodPressure`, `Insulin` and `SkinThickness` are treated as missing and filtered out (`Pregnancies` and `DiabetesPedigreeFunction` keep their zeros, which are valid values). This is a learning exercise, not clinical advice.

## Notebooks

| Notebook | Scope |
|---|---|
| `bmi_glucose_regression.ipynb` (v1) | Single variable (BMI → Glucose), unnormalized gradient descent. Frozen learning log: it never converged (R² ≈ −0.30). Kept as-is on purpose. |
| `bmi_glucose_regression_v2.ipynb` (v2) | Three variables (BMI, Age, BloodPressure → Glucose) with z-score normalization. Converged to cost ≈ 413.5 (MSE ≈ 827). Finding: BMI ≥ 27.5 alone is not a reliable prediabetes rule — age moves predictions more than blood pressure does, and no tested combination reached 100 mg/dL. Residual cost is a model limit (linear, 3 features), not an optimization problem. |
| `bmi_glucose_Logistic_regression_v3.ipynb` (v3) | Regularized logistic regression from scratch (`Outcome` 0/1) with 8 features, 392 rows, z-score, λ sweep and sensitivity sweeps. Train accuracy 0.783 vs 0.668 majority baseline; errors concentrate in the Glucose overlap (median 122 for missed diabetics vs 168.5 for detected). Glucose dominates the learned weights; BloodPressure and Insulin contribute almost nothing. No held-out split yet (v4, after C2). |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Open either notebook with Jupyter and run top to bottom.

## Stack

Python 3.12 · NumPy · pandas · Matplotlib

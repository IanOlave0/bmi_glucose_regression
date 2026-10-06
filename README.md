# BMI–Glucose Linear Regression Lab

Learning record for the Machine Learning Specialization (MLS C1): linear regression implemented from scratch with NumPy, no `sklearn`.

Dataset: Pima Indians Diabetes — https://www.kaggle.com/datasets/mragpavank/diabetes.
Zeros in `BMI`, `Glucose`, `Age`, `BloodPressure` are treated as missing and filtered out. This is a learning exercise, not clinical advice.

## Notebooks

| Notebook | Scope |
|---|---|
| `bmi_glucose_regression.ipynb` (v1) | Single variable (BMI → Glucose), unnormalized gradient descent. Frozen learning log: it never converged (R² ≈ −0.30). Kept as-is on purpose. |
| `bmi_glucose_regression_v2.ipynb` (v2) | Three variables (BMI, Age, BloodPressure → Glucose) with z-score normalization. Converged to cost ≈ 413.5 (MSE ≈ 827). Finding: BMI ≥ 27.5 alone is not a reliable prediabetes rule — age moves predictions more than blood pressure does, and no tested combination reached 100 mg/dL. Residual cost is a model limit (linear, 3 features), not an optimization problem. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Open either notebook with Jupyter and run top to bottom.

## Stack

Python 3.12 · NumPy · pandas · Matplotlib

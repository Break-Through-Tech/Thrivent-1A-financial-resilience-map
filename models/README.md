# Trained models

Fitted scikit-learn pipelines (preprocessing + model) written by the modelling notebooks.

| File | What it is |
|---|---|
| `*.joblib` | The fitted pipeline. **Git-ignored**: regenerate by running the notebook. |
| `*.json` | Selected hyperparameters, CV score, weighted test metrics, feature list and scikit-learn version. Committed. |

## Loading a model

```python
import joblib
import pandas as pd

model = joblib.load("models/gradient_boosting_with_ef.joblib")
data = pd.read_csv("data/processed/model_dataset_2025.csv", low_memory=False)
p_fragile = model.predict_proba(data[features])[:, 1]  # features: see the matching .json
```

Pickled models only load reliably with the scikit-learn version recorded in the `.json`.

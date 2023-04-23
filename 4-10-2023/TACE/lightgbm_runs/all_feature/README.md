# Overview

Notes:
- lightgbm has access to all features
- Evaluated on features extracted from ground truth and predicted segmentations

Ran grid search on lightgbm with the following hyperparameter ranges:
```
lgbm_hyperparams = {
    "learning_rate": (0.1, 0.5, 0.05),
    "max_depth": (1, 10, 1),
    "num_leaves": (10, 100, 5),
    "n_estimators": (10, 100, 5),
}
```
Result (gt):
```
fold 0 MSE: 43.643459795495446, hparams: {'learning_rate': 0.20000000000000004, 'max_depth': 1, 'num_leaves': 10, 'n_estimators': 75}
fold 1 MSE: 51.30750621665585, hparams: {'learning_rate': 0.45000000000000007, 'max_depth': 1, 'num_leaves': 15, 'n_estimators': 95}
fold 2 MSE: 62.27359524706605, hparams: {'learning_rate': 0.1, 'max_depth': 1, 'num_leaves': 10, 'n_estimators': 10}
fold 3 MSE: 49.036060906675466, hparams: {'learning_rate': 0.3500000000000001, 'max_depth': 1, 'num_leaves': 10, 'n_estimators': 75}
fold 4 MSE: 27.054645575183645, hparams: {'learning_rate': 0.45000000000000007, 'max_depth': 2, 'num_leaves': 10, 'n_estimators': 25}
```

Notes:
- lightgbm has access to all features in this experiment

Ran grid search on lightgbm with the following hyperparameter ranges `(min, max, step)`:
```
lgbm_hyperparams = {
    "learning_rate": (0.1, 0.5, 0.05),
    "max_depth": (1, 10, 1),
    "num_leaves": (10, 100, 5),
    "n_estimators": (10, 100, 5),
}
```
Result summary:
```
fold 0 MSE: 43.643459795495446, hparams: {'learning_rate': 0.20000000000000004, 'max_depth': 1, 'num_leaves': 10, 'n_estimators': 75}
fold 1 MSE: 51.30750621665585, hparams: {'learning_rate': 0.45000000000000007, 'max_depth': 1, 'num_leaves': 15, 'n_estimators': 95}
fold 2 MSE: 62.27359524706605, hparams: {'learning_rate': 0.1, 'max_depth': 1, 'num_leaves': 10, 'n_estimators': 10}
fold 3 MSE: 49.036060906675466, hparams: {'learning_rate': 0.3500000000000001, 'max_depth': 1, 'num_leaves': 10, 'n_estimators': 75}
fold 4 MSE: 27.054645575183645, hparams: {'learning_rate': 0.45000000000000007, 'max_depth': 2, 'num_leaves': 10, 'n_estimators': 25}
```
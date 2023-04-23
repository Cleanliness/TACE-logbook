import itertools

import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from lifelines.utils import concordance_index
import datetime
import lightgbm as lgb


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
f1 = [
    'Study ID',
    'original_shape_LeastAxisLength',
    'original_shape_Maximum2DDiameterColumn',
    'original_shape_Maximum2DDiameterRow',
    'original_shape_SurfaceArea',
    'original_gldm_DependenceEntropy',
    'original_glrlm_GrayLevelNonUniformityNormalized',
    'original_glrlm_RunEntropy',
     'original_glszm_LowGrayLevelZoneEmphasis'
]


def load_features(feature_dir):
    df = pd.read_csv(feature_dir)
    cleaned_df = df.select_dtypes(include=numerics)
    imdir = df["Image"].str.split("/")

    cases = []
    for i in imdir:
        case_num = int(i[-1].split("_")[1])
        cases.append(case_num)

    # insert case ID
    cleaned_df.insert(0, "Study ID", cases)
    return cleaned_df


def load_targets(targ_dir):
    df = pd.read_csv(targ_dir)
    return df


def create_mats(input_df, target_df):

    input_cols = input_df.columns.copy()
    target_cols = target_df.columns.copy()

    df = input_df.merge(target_df, on="Study ID")
    X = df.loc[:, input_cols].to_numpy()[:, 1:]
    y = df.loc[:, target_cols].to_numpy()[:, 1]  # using only overall survival for now

    return X, y


def grid_search(hparams, folds, model, log_search=False, track_hist=False):
    """
    Perform grid search over hyperparameters and given training-target set
    for model following scikit-learn-like interface

    :param hparams: dictionary of hyperparameter ranges
    :param train: list of each cross validation fold, each element is: ((train_x, train_y), (val_x, val_y))
    :param model: model implementing a scikit-learn-like interface (fit, set_params)
    :param log_search: True indicates we are using the ranges as exponents to a base of 10
    :param track_hist: True to track complete history of grid search

    Note: use float ranges for negative powers, if using log_search
    """
    # tracking history and best model
    gs_log = {
        "mean_scores": [],
        "models": [],
        "hparams": [],
    }
    best = {
        "score": None,
        "models": None,
        "hparams": None
    }

    # generate all values for grid search
    param_names = hparams.keys()
    param_values = []

    for h in hparams:
        expanded = np.arange(hparams[h][0], hparams[h][1], hparams[h][2]).astype(type(hparams[h][0]))
        if log_search:
            param_values.append(np.power(10, expanded).tolist())
        else:
            param_values.append(expanded.tolist())

    # go over cartesian product of hyperparameters
    for combination in itertools.product(*param_values):
        parameters = {name: value for name, value in zip(param_names, combination)}

        fold_scores = 0
        fold_models = []
        for train, val in folds:
            # TODO change to use general constructor
            model = lgb.LGBMModel(max_depth=1)
            model.set_params(**lgbm_params)

            # set hyperparameters
            model.set_params(**parameters)
            curr = model.fit(*train, eval_set=val)
            fold_models.append(curr)

            # collect metric score
            metric_key = next(iter(curr.best_score_['valid_0']))     # use first given eval metric in model
            fold_scores += curr.best_score_['valid_0'][metric_key]

        curr_avg_score = fold_scores / len(folds)

        # record stats
        if track_hist:
            gs_log["mean_scores"].append(curr_avg_score)
            gs_log["models"].append(fold_models)
            gs_log["hparams"].append(parameters)

        if best["score"] is None or curr_avg_score < best["score"]:
            best["score"] = curr_avg_score
            best["models"] = fold_models
            best["hparams"] = parameters

    return gs_log, best


def cross_val(hparams, features, targets, model):
    """
    Perform cross validation given features, targets and a model
    :param hparams: hyperparameter dictionary to perform grid search on
    :param features: feature matrix
    :param targets: target matrix
    :param model: regression model
    :return:
    """
    kf = KFold(n_splits=5, shuffle=True)
    folds = []
    # grid search over each fold
    for i, (train_idx, val_idx) in enumerate(kf.split(features)):

        # standardize training data
        fold_scaler = StandardScaler()
        X = features[train_idx]
        y = targets[train_idx]
        fold_scaler.fit(X)
        X = fold_scaler.transform(X)

        train_dat = (X, y)
        val_dat = (features[val_idx], targets[val_idx])

        folds.append((train_dat, val_dat))

    return grid_search(hparams, folds, model)


def select_features(feature_names, models):
    """
    Select features with a nonzero feature importance
    """

    sel = None
    for i, m in enumerate(models):
        curr_importances = m.feature_importances_

        if sel is None:
            sel = curr_importances
        else:
            sel += curr_importances

    sel_idx = np.where(sel > 0)

    return dict(zip(feature_names[sel_idx], sel[sel_idx] / len(models)))


def run_lgbm_grid_search(log_dir=""):
    """
    Run full grid search on lgbm model
    """
    lgbm_hyperparams = {
        "learning_rate": (0.01, 0.03, 0.01),
        "max_depth": (1, 10, 1),
        "min_data_in_leaf": (1, 12, 1),
        # "min_sum_hessian_in_leaf": (0.001, 5, 1),
        "num_leaves": (2, 50, 10),
        # "lambda_l1": (0, 5, 0.5),
        # "lambda_l2": (0, 5, 0.5),
        "feature_fraction": (0.4, 1, 0.1),
        # "bagging_fraction": (0.4, 1, 0.1),
    }

    cv_res = cross_val(lgbm_hyperparams, feature_mat, target_mat, lgb.LGBMModel)
    print(cv_res[1])

    save_models(cv_res)

    # getting feature importance dictionary
    sel_features = select_features(feature_names, cv_res[1]["models"])

    # save grid search log
    f = open(f'{log_dir}/grid_search.txt', 'w+')
    f.write(str(cv_res))
    f.close()

    return sel_features


def save_models(cv_res):
    """
    Save models contained in the results of cross validation
    """
    # save model to params
    for i, m in enumerate(cv_res[1]["models"]):
        curr_booster = m.booster_
        curr_booster.save_model(f"fold_{i}_lgbm_model.txt")


if __name__ == '__main__':
    sns.set_theme()
    start_t = datetime.datetime.now()
    # loading pyradiomics features
    pred_dir = r"/home/roy/Documents/TACE/gt_features.csv"
    td = r"/home/roy/Documents/TACE/os_pfs_nopass.csv"
    feature_df = load_features(pred_dir)
    targ_df = load_targets(td)
    feature_names = np.array(feature_df.columns)[1:]

    # using selected feature sets
    feature_df = feature_df[f1]
    selected_df = targ_df
    feature_mat, target_mat = create_mats(feature_df, selected_df)

    lgbm_params = {
        "metric": ["mse"],
        "objective": "regression",
        "early_stopping_rounds": 100,
        "num_iterations": 500,
        "max_depth": -1,
        # "min_data_in_leaf": 2,
    }

    print(run_lgbm_grid_search())
    print(start_t, datetime.datetime.now())
    print("lasso gt features selected from single run, training lightgbm on gt features")
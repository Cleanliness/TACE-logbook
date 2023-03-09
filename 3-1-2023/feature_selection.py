import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso, LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


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


def lasso_cross_val(hparams, features, targets):
    """
    Perform 5-fold cross validation over all alpha values stored in hparams.
    """

    kf = KFold(n_splits=5)
    # grid search over alpha
    scores = []
    l1_norms = []
    best = float("inf")
    best_ensemble = None

    for alph in hparams:
        curr_scores = []
        curr_l1 = []
        models = []
        # compute one pass over a fold
        for i, (train_idx, val_idx) in enumerate(kf.split(features)):

            # standardize training data
            fold_scaler = StandardScaler()
            X = features[train_idx]
            y = targets[train_idx]
            fold_scaler.fit(X)
            X = fold_scaler.transform(X)

            # train model
            reg = Lasso(alpha=alph, max_iter=1000).fit(X, y)
            models.append(reg)

            # get validation MSE and metrics
            X_val = fold_scaler.transform(features[val_idx])

            y_val = targets[val_idx]
            y_pred = reg.predict(X_val)
            curr_scores.append(mean_squared_error(y_val, y_pred)/2)
            curr_l1.append(np.linalg.norm(reg.coef_, ord=1))

        scores.append(curr_scores)
        l1_norms.append(curr_l1)

        # update best models
        if np.mean(curr_scores) < best:
            best = np.mean(curr_scores)
            best_ensemble = models
    # convert result to numpy
    return np.array(scores), np.array(l1_norms)


if __name__ == '__main__':
    sns.set_theme()

    # loading pyradiomics features
    pred_dir = r"/home/roy/Documents/TACE/predicted_features.csv"
    td = r"/home/roy/Documents/TACE/os_pfs_nopass.csv"
    feature_df = load_features(pred_dir)
    targ_df = load_targets(td)

    # create input and label matrices, split into folds
    feature_mat, target_mat = create_mats(feature_df, targ_df)

    # start searching alphas by order of magnitudes
    alphas = np.arange(0.1, 10, 0.001)
    scores, norms = lasso_cross_val(alphas, feature_mat, target_mat)

    h = np.mean(scores, axis=1)
    # visualization
    xlabels = np.array([[i for j in range(5)] for i in alphas]).flatten()
    sns.scatterplot(xlabels, scores.flatten())
    plt.xlabel("log_10(alpha)")
    plt.ylabel("MSE")
    plt.savefig("log_alpha.png")

    b = np.argmin(scores)
    bbb = 3
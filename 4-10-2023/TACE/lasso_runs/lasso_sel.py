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

    # drop study_id column in X
    df = input_df.merge(target_df, on="Study ID")
    X = df.loc[:, input_cols].to_numpy()[:, 1:]
    y = df.loc[:, target_cols].to_numpy()[:, 1]  # using only overall survival for now

    return X, y


def lasso_cross_val(hparams, features, targets):
    """
    Perform 5-fold cross validation over all alpha values stored in hparams.
    """

    kf = KFold(n_splits=5, shuffle=True)
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
            curr_scores.append(mean_squared_error(y_val, y_pred))
            curr_l1.append(np.linalg.norm(reg.coef_, ord=1))

        scores.append(curr_scores)
        l1_norms.append(curr_l1)

        # update best models
        if np.mean(curr_scores) < best:
            best = np.mean(curr_scores)
            best_ensemble = models

    # convert result to numpy
    return np.array(scores), np.array(l1_norms), best_ensemble


def get_selected_features(models, feature_col_names):
    """
    Get features selected by lasso_runs regression model
    :return: list of selected feature names and their indices
    """
    res = np.array([], dtype=int)
    for i in models:
        res = np.append(res, np.where(np.abs(i.coef_) > 0))

    res = np.unique(res)
    return feature_col_names[res], res


def visualize_gs_results(alphs, scores):
    """
    Save results of grid search as a graph
    """
    xlabels = np.array([[i for j in range(5)] for i in alphs]).flatten()
    sns.scatterplot(xlabels, scores.flatten())
    plt.xlabel("alpha")
    plt.ylabel("MSE")
    plt.savefig("lasso_gridsearch.png")


def count_features(feature_dict, selected):
    for s in selected:
        if s not in feature_dict:
            feature_dict[s] = 0
        feature_dict[s] += 1

    return feature_dict


if __name__ == '__main__':

    # loading pyradiomics features
    feature_dir = r"/home/roy/Documents/TACE/gt_features.csv"
    td = r"/home/roy/Documents/TACE/os_pfs_nopass.csv"
    feature_df = load_features(feature_dir)
    targ_df = load_targets(td)
    feature_names = np.array(feature_df.columns)[1:]

    # create input and label matrices, split into folds
    feature_mat, target_mat = create_mats(feature_df, targ_df)
    tot_scores = np.array([])
    tot_alpha = np.array([])

    # run 5-fold CV n times, see what features selected
    feature_dict = {}
    cv_counter = 0
    while cv_counter < 100:
        # start searching alphas near 1.95
        alphas = np.arange(1.9, 2, 0.001)
        scores, norms, best_models = lasso_cross_val(alphas, feature_mat, target_mat)

        sel_features, sel_idx = get_selected_features(best_models, feature_names)
        count_features(feature_dict, sel_features)

        # report performance
        best_alpha_idx = np.argmin(np.mean(scores, axis=1))
        best_scores = scores[best_alpha_idx]
        tot_scores = np.append(tot_scores, best_scores)
        tot_alpha = np.append(tot_alpha, alphas[best_alpha_idx])
        print(f"score: {np.mean(best_scores)} ± {np.std(best_scores)}, alpha: {alphas[best_alpha_idx]}")

        print(f"feature selection {cv_counter} completed")
        cv_counter += 1

    feature_dict = {k: v for k, v in sorted(feature_dict.items(), key=lambda item: item[1])}
    print("============== final_results ==============")
    print(f"score: {np.mean(tot_scores)} ± {np.std(tot_scores)}, alpha: {np.mean(tot_alpha)} ± {np.std(tot_alpha)}")
    print(feature_dict)

    # plot selected feature frequency
    plt.bar(feature_dict.keys(), feature_dict.values())
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()    # adjust figure size
    plt.savefig("feature_frequency.png")





from utils import *
import numpy as np

if __name__ == '__main__':

    # loading pyradiomics features
    feature_dir = r"/home/roy/Documents/TACE/predicted_features.csv"
    td = r"/home/roy/Documents/TACE/os_pfs_nopass.csv"
    feature_df = load_features(feature_dir)
    targ_df = load_targets(td)
    feature_names = np.array(feature_df.columns)[1:]

    # create input and label matrices, split into folds
    feature_mat, target_mat = create_mats(feature_df, targ_df)

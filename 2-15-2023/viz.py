import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import kl_div
import numpy as np


def plot_features(df):
    """
    Plot distribution of each radiomic feature
    :param df: dataframe containing predicted and ground truth features
    """
    for c in df.columns:
        curr_df = df[[c, "class"]]
        sns.catplot(data=curr_df, kind='violin', x="class", y=c)
        plt.savefig(rf"C:\Users\royga\Desktop\pr_results\feature_viz\{c}.png")
        print(f"finished {c}")


def plot_kl_div(df1, df2):
    """
    Plot histogram of KL divergences between ground truth and predicted
    features
    :param df1: ground truth dataframe
    :param df2: predicted dataframe
    :return:
    """
    kl_divs = []
    for c in df1.columns[:-1]:
        p1 = np.histogram(df1[c].values, bins=6)[0]
        p1 = p1 / np.sum(p1)
        p2 = np.histogram(df2[c].values, bins=6)[0]
        p2 = p2 / np.sum(p2)

        kl_divs.append(sum(kl_div(p1, p2)))

    # replace inf with -1 for easier viewing
    kl_divs = np.array(kl_divs)
    kl_divs[kl_divs == np.infty] = -1
    sns.histplot(kl_divs)
    plt.show()


if __name__ == '__main__':
    # Apply the default theme
    sns.set_theme()
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    # loading pyradiomics features
    gt_df = pd.read_csv(r"C:\Users\royga\Desktop\pr_results\tace_gt_results.csv")
    gt_df = gt_df.select_dtypes(include=numerics)
    gt_df["class"] = ["Ground Truth" for i in range(gt_df.shape[0])]

    pred_df = pd.read_csv(r"C:\Users\royga\Desktop\pr_results\tace_stitch_results.csv")
    pred_df = pred_df.select_dtypes(include=numerics)
    pred_df["class"] = ["Prediction" for i in range(pred_df.shape[0])]

    # get KL divergence histogram
    plot_kl_div(gt_df, pred_df)

    # uncomment to save plot of each feature distribution
    # df = pd.concat((gt_df, pred_df), axis=0)
    # plot_features(df)

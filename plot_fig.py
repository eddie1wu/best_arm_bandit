import numpy as np
import pandas as pd

from scipy.stats import beta

import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from utils import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def plot_comparison(file1, file2, file3, save_name):

    with open(result_path + file1, "rb") as f:
        left = pickle.load(f)
    with open(result_path + file2, "rb") as f:
        mid = pickle.load(f)
    with open(result_path + file3, "rb") as f:
        right = pickle.load(f)

    right = np.array(right)

    dot_size = 15
    lw = 0.7
    fig, ax = plt.subplots(1, 3, figsize = (20, 6))
    ax[0].scatter(range(len(left)), left, color = 'black', s = dot_size, alpha = 0.4)
    ax[0].scatter(range(5), left[:5], color = 'red', s = 25)
    ax[0].set_title("(a) Built-in feature importance")
    ax[0].set_xlabel("Feature index")
    ax[0].set_ylabel("Importance")
    ax[0].grid(True)

    ax[1].scatter(range(len(mid)), mid, color = 'black', s = dot_size, alpha = 0.4)
    ax[1].scatter(range(5), mid[:5], color = 'red', s = 25)
    ax[1].set_title("(b) Permutation importance")
    ax[1].set_xlabel("Feature index")
    ax[1].set_ylabel("Importance")
    ax[1].grid(True)

    for i in range(right.shape[1] - 1, -1, -1):
        if i < 5:
            ax[2].plot(range(right.shape[0]), right[:, i], color = 'red', alpha = 1, lw= 1.2)
        else:
            ax[2].plot(range(right.shape[0]), right[:, i], color = 'black', alpha=0.4, lw = lw)
    ax[2].set_title("(c) Bandit importance")
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("Inclusion probability")
    ax[2].grid(True)
    ax[2].set_yticks(np.arange(0, 1.1, 0.1))

    plt.tight_layout()
    plt.savefig(fig_path + save_name, dpi=300)


def plot_convergence(df1, save_name):

    means = [np.mean(df1.iloc[0, :]), np.mean(df1.iloc[1, :])]
    prob = [1 - np.mean(df1.iloc[0, :] < 500), 1 - np.mean(df1.iloc[1, :] < 500)]
    algos = ["Thompson", "Top-two Thompson"]

    bar_width = 0.25
    x = np.arange(len(algos))

    fig, (ax, his) = plt.subplots(1, 2, figsize = (20,8))

    # Plot bar chart comparison
    ax.bar(x-bar_width/2, means, color = "black", width = bar_width, label = "Mean iterations", alpha = 1)
    ax.set_ylabel("Mean iterations to convergence", fontsize = 14, color = "black")
    ax.tick_params(axis = "y")
    ax.spines["top"].set_visible(False)

    ax2 = ax.twinx()
    ax2.bar(x+bar_width/2, prob, color = "red", width = bar_width, label = "Probability", alpha = 0.8)
    ax2.set_ylabel("Probability of slow convergence", fontsize = 14, color = "red", rotation = -90, labelpad = 20)
    ax2.set_ylim(0,1)
    ax2.tick_params(axis = "y", colors = "red")
    ax2.spines['right'].set_color('red')
    ax2.spines["top"].set_visible(False)

    ax.set_xticks(x)
    ax.set_xticklabels(algos, fontsize = 14)

    # Plot histograms
    sns.histplot(df1.iloc[0, :], kde = True, bins = 10, alpha = 1, label = "Thompson sampling", color = "black", ax = his)
    sns.histplot(df1.iloc[1, :], kde = True, bins = 10, alpha = 0.5, label = "Top-two sampling", color = "red", ax = his)
    his.set_ylabel("Density", fontsize = 14)
    his.set_xlabel("Number of iterations", fontsize = 14)
    his.legend()

    plt.tight_layout()
    plt.savefig(fig_path + save_name, dpi = 300)


def plot_online(save_name):

    results = []
    with open(result_path + "online_gb.pkl", "rb") as f:
        results.append(pickle.load(f))
    with open(result_path + "online_mlp.pkl", "rb") as f:
        results.append(pickle.load(f))
    with open(result_path + "online_rf.pkl", "rb") as f:
        results.append(pickle.load(f))

    fig, ax = plt.subplots(1, 3, figsize = (20,8))
    for k in range(3):
        result = results[k]
        result = np.array(result)
        for i in range(result.shape[1] - 1, -1, -1):
            if i < 5:
                ax[k].plot(range(result.shape[0]), result[:, i], color = 'red', alpha = 1, lw= 1.2)
            else:
                ax[k].plot(range(result.shape[0]), result[:, i], color = 'black', alpha=0.4, lw = 0.7)
        ax[k].set_xlabel("Time")
        ax[k].set_ylabel("Inclusion probability")
        ax[k].grid(True)
        ax[k].set_yticks(np.arange(0, 1.1, 0.1))

    ax[0].set_title("(a) Gradient boosting regressor", fontsize = 15)
    ax[1].set_title("(b) Fully connected neural network", fontsize = 15)
    ax[2].set_title("(c) Random forest regressor", fontsize = 15)

    plt.tight_layout()
    plt.savefig(fig_path + save_name, dpi = 300)


def plot_corrmatrix():

    df = pd.read_csv(data_path + "gkx_subset.csv")
    df['date'] = pd.PeriodIndex(df['date'], freq='M')
    start = pd.Period("2010-01", freq = "M")
    end = pd.Period("2019-12", freq = "M")
    subset = df.loc[(df['date']>= start) & (df['date']<= end), df.columns[3:96]]
    print("Subsetted")

    corr_matrix = subset.corr()

    print("Plot corr matrix")
    plt.figure(figsize = (22, 18))
    sns.heatmap(corr_matrix, cmap = "coolwarm", annot = False, fmt = ".2f", linewidth = 0.5, vmin = -1, vmax = 1)
    plt.title("Correlation matrix of 94 firm-level features", fontsize = 20)

    plt.tight_layout()
    plt.savefig(fig_path + "corr_matrix.png", dpi = 300)


def plot_gkx():
    file1 = "gkx_probabilities.pkl"
    file2 = "gkx_colnames.pkl"
    save_name = "gkx.png"

    with open(result_path + file1, "rb") as f:
        probabilities = pickle.load(f)

    with open(result_path + file2, "rb") as f:
        colnames = pickle.load(f)

    feature_names = np.array(colnames[:94])
    probabilities = np.array(probabilities)

    features = probabilities[:, :94]
    noise = probabilities[:, (94+74):]

    print(features.shape)
    print(noise.shape)

    df = pd.DataFrame({'feature': feature_names, 'probability': features[-1, :]})
    df = df.sort_values(by = 'probability', ascending = False).reset_index(drop = True)

    # Plot here
    fig, ax = plt.subplots(1,2, figsize = (12,6))

    selected_cols = np.random.choice(noise.shape[1], 1000, replace = False)
    noise = noise[:, selected_cols]

    # Left panel, convergence
    for i in range(noise.shape[1]):
        ax[0].plot(range(noise.shape[0]), noise[:, i], color='black', alpha=0.4, lw=0.7, label = "Noise")

    for i in range(features.shape[1]):
        ax[0].plot(range(features.shape[0]), features[:, i], color='red', alpha=0.4, lw=0.7, label = "Features")
    ax[0].set_title("(a) Inclusion probabilities over time", fontsize = 14)
    ax[0].set_xlabel("Time", fontsize = 14)
    ax[0].set_ylabel("Inclusion probability", fontsize = 14)
    ax[0].grid(True)
    ax[0].set_yticks(np.arange(0, 1.1, 0.1))
    handles, labels = ax[0].get_legend_handles_labels()
    unique_legend = dict(zip(labels, handles))
    ax[0].legend(unique_legend.values(), unique_legend.keys(), loc="upper right")

    # Right panel, bar chart comparison
    num_features = 25
    ax[1].bar(df.loc[:num_features, "feature"], df.loc[:num_features, "probability"], color = 'red', alpha = 0.7)
    ax[1].set_title(f"(b) Top {num_features} features inclusion probabilities ", fontsize = 14)
    ax[1].set_ylim(0.4, 0.85)
    ax[1].set_xticks(range(len(df.loc[:num_features, "feature"])))
    ax[1].set_xticklabels(df.loc[:num_features, "feature"], rotation = 65, ha = "right")
    ax[1].grid(True, axis = 'y', alpha = 0.7)

    plt.tight_layout()
    plt.savefig(fig_path + save_name, dpi = 300)


if __name__ == "__main__":

    data_path = "/Users/eddiewu/Downloads/feature_bandit/"
    result_path = "/Users/eddiewu/Downloads/feature_bandit/plot_results/"
    fig_path = "figures/"

    # Compare to model specific methods
    plot_comparison("rf_importance_1.pickle",
                    "rf_pi_1.pickle",
                    "rf_probabilities_1.pkl",
                    "rf_friedman1.png")

    plot_comparison("rf_importance_2.pickle",
                    "rf_pi_2.pickle",
                    "rf_probabilities_2.pkl",
                    "rf_friedman2.png")

    # Compare convergence
    plot_convergence(pd.read_csv(result_path + "convergence_stats_rf.csv"),
                     "convergence_1.png")

    plot_convergence(pd.read_csv(result_path + "convergence_stats_lasso.csv"),
                     "convergence_2.png")

    # Demonstrate online learning
    plot_online("online_learning.png")

    # Application to gkx
    plot_corrmatrix()
    plot_gkx()
# Learning feature importance with combinatorial multi-armed bandit

This repository hosts my codes for the project on improving the accuracy and efficiency of learning feature importance in high-dimensional settings with combinatorial multi-armed bandits.

Abstract: Understanding variable importance is a critical task in statistical modelling, especially in high-dimensional data analysis. Building on existing combinatorial bandit approaches, this paper proposes a novel pure exploration bandit-based method for learning feature importance in a model-agnostic manner. The iterative algorithm is inspired by best-arm identification and integrates permutation importance as a model-agnostic reward mechanism. Through numerical simulations, I show that the approach achieves higher accuracy and faster convergence compared to existing methods, and can adapt to both offline and online settings. Additionally, I apply my framework to empirical asset pricing and show that it recovers key predictors of stock returns while efficiently filtering out noises. My result suggests that the proposed method provides a robust, interpretable and computationally scalable solution for learning feature importance.

## File structure

The `figures` folder contains the figures in the paper.

[`BinaryBandit.py`](BinaryBandit.py) the Bernoulli bandit class.

[`data_clean.py`](data_clean.py) data cleaning step, after fetching and merging.

[`data_fetch.py`](data_fetch.py) fetch returns from CRSP.

[`data_merge.py`](data_merge.py) merge firm level covariates of Gu, Kelly and Xiu (2020) with excess returns, after fetching.

[`data_preprocess.py`](data_preprocess.py) preprocessing steps for the data, after fetching, merging and cleaning.

[`ImportanceBandit.py`](ImportanceBandit.py) contains the class for bandit feature selection.

[`MLP.py`](MLP.py) contains the most basic MLP class, in Pytorch.

[`plot_fig.py`](plot_fig.py) plots all the figures in the paper.

[`run_best_arm.py`](run_best_arm.py) compares the convergence rates and variances of difference methods of best-arm identification under a fixed confidence setting.

[`run_convergence.py`](run_convergence.py) compares the convergence rates of Thompson sampling to top-two Thompson sampling.

[`run_gkx.py`](run_gkx.py) run bandit feature selection on the dataset of Gu, Kelly and Xiu (2020).

[`run_online.py`](run_online.py) runs different datasets in an online setting.

[`run_predictive_check.py`](run_predictive_check.py) runs sanity checks comparing the predictive performances of models using high-dimensional covariates.

[`run_rf.py`](run_rf.py) compares bandit feature importance to random forest impurity based importance and permutation importance.

[`Sampler.py`](Sampler.py) the Sampler class which contains all the best arm identification algorithms.

[`utils.py`](utils.py) contains the utility functions for generating data and plotting.

# Learning feature importance with combinatorial multi-armed bandit

This repository hosts my codes for the project on improving the accuracy and efficiency of learning feature importance in high-dimensional settings with combinatorial multi-armed bandits.

Abstract: Understanding variable importance is a critical task in statistical modelling, especially in high- dimensional data analysis. Traditional methods such as LASSO and tree-based approaches tend to struggle with accuracy, robustness and computational efficiency in feature selection. Building on existing combinatorial bandit frameworks, this paper proposes a novel pure exploration bandit- based method for learning feature importance in a model-agnostic manner. The iterative algorithm is inspired by best-arm identification and integrates permutation importance as a model-agnostic reward mechanism. Numerical simulations show that the approach achieves higher accuracy and faster convergence compared to existing methods, and can adapt to both offline and online settings. Additionally, applying the framework to empirical asset pricing demonstrates that it recovers key predictors of stock returns while efficiently filtering out noises. These results suggest that the proposed method provides a robust, interpretable and computationally scalable solution for learning feature importance.

Running order:

-   Numerical simulations: run_rf.py -\> run_convergence.py -\> run_online.py

-   Empirical application: data_fetch.py -\> data_merge.py -\> data_clean.py -\> data_preprocess.py -\> run_gkx.py

## File structure

The `figures` folder contains the figures in the paper.

[`BinaryBandit.py`](BinaryBandit.py) the Bernoulli bandit class.

[`data_clean.py`](data_clean.py) data cleaning step, after fetching and merging.

[`data_fetch.py`](data_fetch.py) fetch returns from CRSP.

[`data_merge.py`](data_merge.py) merge firm level covariates of Gu, Kelly and Xiu (2020) with excess returns, after fetching.

[`data_preprocess.py`](data_preprocess.py) preprocessing steps for the data, after fetching, merging and cleaning.

[`ImportanceBandit.py`](ImportanceBandit.py) contains the class for bandit feature selection.

[`MLP.py`](MLP.py) contains the most basic MLP class, in Pytorch.

[`plot_fig.py`](plot_fig.py) plots all the figures in the paper. But need to generate results using the other scripts first.

[`run_best_arm.py`](run_best_arm.py) compares the convergence rates and variances of difference methods of best-arm identification under a fixed confidence setting for stochastic MAB.

[`run_convergence.py`](run_convergence.py) compares the convergence rates of Thompson sampling to top-two Thompson sampling.

[`run_gkx.py`](run_gkx.py) run bandit feature selection on the dataset of Gu, Kelly and Xiu (2020).

[`run_online.py`](run_online.py) runs different datasets in an online setting.

[`run_predictive_check.py`](run_predictive_check.py) runs sanity checks comparing the predictive performances of models using high-dimensional covariates.

[`run_rf.py`](run_rf.py) compares bandit feature importance to random forest impurity based importance and permutation importance.

[`Sampler.py`](Sampler.py) the Sampler class which contains all the best arm identification algorithms.

[`utils.py`](utils.py) contains the utility functions for generating data and plotting.

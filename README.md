# Best Arm Identification for Variable Selection

This repository hosts my codes for the project on exploring ways of improving variable selection using multi-armed bandits, building on Rockova & Liu 2021.

Here is the abstract: this research explores variable selection in a high-dimensional setting through an application of best-arm identification algorithms. Building on the framework of Rockova & Liu 2021, this research demonstrates how combinatorial bandits can be welded to algorithms such as LASSO and random forest to improve the accuracy of variable selection compared to running such algorithms directly on high-dimensional data. This research also augments the existing Thompson variable selection framework by integrating top-two Thompson sampling and bootstrap aggregation, which are shown to yield robust results and improve on accuracy and variance of variable selection in numerical experiments. Future work aims to explore the theoretical reasons to why these algorithms work and prove properties such as regret bound and consistency.

## File structure

The `figures` folder contains the analysis plots comparing methods of variable selection based on the accuracy of selecting the important variables.

[`BinaryBandit.py`](BinaryBandit.py) the Bernoulli bandit class.

[`MLP.py`](MLP.py) the feed-forward neural network class.

[`run_best_arm.py`](run_best_arm.py) compares the convergence rates and variances of difference methods of best-arm identification under a fixed confidence setting.

[`run_predictive_check.py`](run_predictive_check.py) runs sanity checks comparing the predictive performances of models under high-dimensional covariates.

[`run_variable_selection.py`](run_variable_selection.py) runs variable selection using Thompson sampling + LASSO or Thompson sampling + Random Forest.

[`Sampler.py`](Sampler.py) the Sampler class which contains all the best arm identification algorithms.

[`ThompsonForest.py`](ThompsonForest.py) Thompson sampling with random forest as the model which generates rewards at each time step.

[`ThompsonLasso.py`](ThompsonLasso.py) Thompson sampling with LASSO as the model which generates rewards at each time step.

[`utils.py`](utils.py) contains the utility functions used in the best arm identification convergence experiments, and the best arm variable selection experiments.

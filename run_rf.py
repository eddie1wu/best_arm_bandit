import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

from ImportanceBandit import ImportanceBandit
from utils import *

import pickle


def compare_methods():



    # Prepare data
    n = 300
    p = 1000
    dgp = 2
    X, y = make_friedman(
        n_samples = n,
        n_features = p,
        std_error = 1,
        dgp = dgp,
        random_state = 11
    )

    result_path = "/Users/eddiewu/Downloads/feature_bandit/plot_results/"


    ### Fine tune and train random forest
    # Define param grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'max_features': [None, 'sqrt']
    }

    # Perform grid search
    rf = RandomForestRegressor(random_state = 18)
    grid_search = GridSearchCV(estimator = rf,
                               param_grid = param_grid,
                               scoring = 'neg_mean_squared_error',
                               cv = 5,
                               verbose = 2,
                               n_jobs = -1)
    grid_search.fit(X, y)

    # Get the best model
    best_model = grid_search.best_estimator_
    print(f"The optimal params are {grid_search.best_params_}")

    # Obtain the built-in importance
    rf_importance = best_model.feature_importances_

    with open(result_path + f"rf_importance_{dgp}.pickle", "wb") as f:
        pickle.dump(rf_importance, f)

    # plt.scatter(range(len(rf_importance)), rf_importance)
    # plt.show()

    # Obtain permutation importance
    X_test, y_test = make_friedman(
        n_samples = 100,
        n_features = p,
        std_error = 1,
        dgp = dgp,
        random_state = 6
    )

    result = permutation_importance(
        best_model,
        X_test,
        y_test,
        n_repeats=10,
        n_jobs=-1
    )

    importance_score = result.importances_mean

    with open(result_path + f"rf_pi_{dgp}.pickle", "wb") as f:
        pickle.dump(importance_score, f)

    # plt.scatter(range(len(importance_score)), importance_score)
    # plt.show()

    ### Bandit importance
    alpha_prior = np.ones(p)
    beta_prior = np.ones(p)

    model = RandomForestRegressor(
        n_estimators = 100,
        max_depth = 10,
        max_features = None,
        n_jobs = -1
    )

    agent = ImportanceBandit(
        model,
        alpha_prior,
        beta_prior,
        max_features = 1
    )

    num_epochs = 300
    probabilities, time, r2_list = agent.train(
        X,
        y,
        num_epochs,
        method = 'ttts',
        how_reward = 'absolute',
        threshold = 0.01,
        verbose = 10)

    with open(result_path + f"rf_probabilities_{dgp}.pkl", "wb") as f:
        pickle.dump(probabilities, f)


if __name__ == '__main__':

    compare_methods()

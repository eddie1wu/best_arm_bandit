import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sympy.vector import Gradient

from tqdm import tqdm

import pickle

from ImportanceBandit import ImportanceBandit

from utils import *

def train_online(data, model, n_samples, n_features, n_informative, num_iter, plot = False, **kwargs):

    # Create the bandit for variable importance
    alpha_prior = np.ones(n_features)
    beta_prior = np.ones(n_features)

    selector = ImportanceBandit(model, alpha_prior, beta_prior, max_features = 0.5, bootstrap = False, time_series = False)

    probabilities = []
    r2_list = []

    for t in tqdm(range(num_iter)):
        X, y = data(
            n_samples=n_samples,
            n_features=n_features,
            **kwargs
        )

        probability, _, r2 = selector.train(
            X,
            y,
            num_epochs=1,
            method='ttts',
            how_reward='relative',
            threshold=0.01,
            verbose=0,
            disable_progress=True
        )
        probabilities.append(probability[0])
        r2_list.append(r2[0])

    if plot:
        # Plot posterior probabilities
        plot_posterior_mean(probabilities, num_iter, n_informative,
                            title = 'Posterior probabilities over time',
                            save_name = 'figures/best_m_lasso.png')

        print(r2_list)
        print(len(r2_list))
        print(num_iter)

        plt.scatter(range(num_iter), r2_list)
        plt.show()

        print(probabilities[-1])

    return probabilities


if __name__ == '__main__':

    result_path = "/Users/eddiewu/Downloads/feature_bandit/plot_results/"

    n = 300
    p = 500
    q = 5
    num_iter = 50

    # model = GradientBoostingRegressor(
    #     n_estimators = 100,
    #     max_depth = 3,
    #     min_samples_split = 4,
    #     n_iter_no_change = 10
    # )

    # model = RandomForestRegressor(
    #     n_estimators=10,
    #     max_depth=4,
    #     min_samples_split=2,
    #     n_jobs=-1
    # )

    model = MLPRegressor(
        hidden_layer_sizes = (64, 32),
        learning_rate = 'adaptive',
        learning_rate_init = 0.005,
        max_iter = 500,
        early_stopping = True,
        validation_fraction = 0.1,
        n_iter_no_change = 10
    )

    probabilities = train_online(make_friedman, model, n, p, q, num_iter, plot = True, std_error = 1, dgp = 2)

    with open(result_path + "online_mlp.pkl", "wb") as f:
        pickle.dump(probabilities, f)


import numpy as np
import pandas as pd

# from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from tqdm import tqdm

import pickle

from ImportanceBandit import ImportanceBandit
from utils import *

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def run_data(df):

    # Preprocessing
    df = df.drop(df.columns[-8:], axis=1) # Drop the macro variables
    df = df.sort_values(by = ['date', 'permno'])
    df['date'] = pd.PeriodIndex(df['date'], freq = 'M')

    n_industries = df['sic2'].nunique()
    df = pd.get_dummies(df, columns = ['sic2'])
    print(f"The number of unique industries is {n_industries}.")

    # Define quantities
    noise_dim = 5000
    p = 94 + 74 + noise_dim # 94 firm features, 74 industries

    model = HistGradientBoostingRegressor(
        learning_rate = 0.05,
        max_iter = 200,
        max_leaf_nodes = 50,
        max_depth = 10,
        min_samples_leaf = 20,
        l2_regularization = 0.01,
        max_features = 0.5,
        early_stopping = True
    )

    # Create the bandit for variable importance
    num_iter = 120 # 10 years
    train_interval = 1
    start = pd.Period("2010-01", freq = 'M')
    end = start + train_interval

    alpha_prior = np.ones(p)
    beta_prior = np.ones(p)
    selector = ImportanceBandit(model, alpha_prior, beta_prior, max_features = 0.1, bootstrap = True, time_series = False)

    probabilities = []
    r2_list = []

    for t in tqdm(range(num_iter)):

        print(f"Running on period: {start} to {end}")

        temp = df.loc[ (df['date'] >= start) & (df['date'] < end), df.columns[2:] ].to_numpy(dtype = float)
        X = temp[:, 1:]
        y = temp[:, 0]

        X_noise = np.random.randn(X.shape[0], noise_dim)
        X = np.hstack((X, X_noise))

        print(f"The shape of X is {X.shape}")

        probability, _, r2 = selector.train(
            X,
            y,
            num_epochs = 5,
            method = 'ttts',
            how_reward = 'relative',
            threshold = 0.02,
            verbose = 0,
            disable_progress = True
        )
        probabilities += probability
        r2_list += r2

        start += 1
        end += 1

    with open(result_path + "gkx_probabilities.pkl", "wb") as f:
        pickle.dump(probabilities, f)

    with open(result_path + "gkx_r2.pkl", "wb") as f:
        pickle.dump(r2_list, f)

    with open(result_path + "gkx_colnames.pkl", "wb") as f:
        pickle.dump(df.columns[3:].to_list(), f)


def tune_hyperparam(df):

    param_grid = {
        'learning_rate': [0.005, 0.01, 0.05, 0.1],
        'max_iter': [200, 500],
        'max_leaf_nodes': [31, 50],
        'max_depth': [5, 10, 15],
        'min_samples_leaf': [10, 20, 30],
        'l2_regularization': [0.01, 0.1, 0.2],
        'max_features': [0.5, 1.0]
    }

    model = HistGradientBoostingRegressor(
        validation_fraction = 0.15,
        early_stopping = True
    )

    grid_search = GridSearchCV(
        estimator = model,
        param_grid = param_grid,
        scoring = 'neg_mean_squared_error',
        cv = 5,
        n_jobs = -1,
        verbose = 2
    )

    date = pd.Period("2000-01", freq = 'M')
    temp = df.loc[ df['date'] == date, df.columns[2:]].to_numpy(dtype = float)

    X = temp[:, 1:]
    y = temp[:, 0]

    print(X.shape)
    print(y.shape)

    grid_search.fit(X, y)

    print(f"The best params are {grid_search.best_params_}")
    print(f"The best score is {grid_search.best_score_}")


if __name__ == '__main__':

    # Options
    do_tuning = False

    # Define paths
    data_path = "/Users/eddiewu/Downloads/feature_bandit/"
    result_path = "/Users/eddiewu/Downloads/feature_bandit/plot_results/"

    # Load data
    file = 'gkx_subset.csv'
    df = pd.read_csv(data_path + file)

    # Hyperparam tuning
    if do_tuning:
        tune_hyperparam(df)

    run_data(df)

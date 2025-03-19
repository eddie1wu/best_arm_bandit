import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso

from tqdm import tqdm

from ImportanceBandit import ImportanceBandit

from utils import *

result_path = "/Users/eddiewu/Downloads/feature_bandit/"

def train_once(method, plot = False, seed = 34):

    n = 300
    p = 500
    q = 5

    # Make Friedman data
    X, y = make_friedman(n, p, 1, dgp = 2, random_state = seed)

    alpha_prior = np.ones(p)
    beta_prior = np.ones(p)

    model = Lasso(alpha = 0.1, max_iter = 10000)

    selector = ImportanceBandit(model, alpha_prior, beta_prior, max_features = 1, bootstrap = True)

    true_features = np.arange(0, q)

    probabilities, time, r2_list = selector.train(
        X,
        y,
        method = method,
        how_reward = 'relative',
        threshold = 0.04,
        features_informative = true_features,
        max_iter = 500,
        verbose = 0,
        disable_progress = True
    )

    if plot:
        print(probabilities[-1])
        plot_posterior_mean(probabilities, time, q,
                            title='Posterior probabilities over time',
                            save_name='figures/best_m_lasso.png')

        plt.scatter(range(time), r2_list)
        plt.show()

    return time

num_sim = 50
ts_time = []
ttts_time = []
for t in tqdm(range(num_sim)):
    ts_time.append(train_once('ts', seed = t))
    ttts_time.append(train_once('ttts', seed = t))

    if t%2 == 0:
        print(f"At the current time step {t}, we have {ts_time[-6:]} \n and {ttts_time[-6:]}.")

plt.scatter(range(num_sim), ts_time, color='blue')
plt.scatter(range(num_sim), ttts_time, color='red')
plt.show()


series1 = pd.Series(ts_time)
series2 = pd.Series(ttts_time)

# Combine data into a list
data = [series1, series2]

temp = pd.DataFrame(data)
temp.to_csv(result_path + 'plot_results/convergence_stats.csv')

# Create the box plot
plt.boxplot(data, tick_labels=["TS", "TTTS"])
plt.show()


# train_once('ttts', plot = True, seed = 2)
# train_once('ts', plot = True, seed = 2)

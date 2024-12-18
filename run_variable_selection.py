import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, Lasso
from tqdm import tqdm

import matplotlib.pyplot as plt

from ThompsonLasso import ThompsonLasso
from utils import *

# Generate data
p = 300
n = 300
sigma = 0.5
q = 5
X, y = gen_linear_data(p, n, q, sigma, random_state = 4)

# Standardize data to run LASSO
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X = (X - mean) / std

# Create the Thompson sampler
alpha_prior = np.ones(p)
beta_prior = np.ones(p)
C = (np.sqrt(5) - 1) / 2
selector = ThompsonLasso(alpha_prior, beta_prior, C, threshold = 0.02)

# Train the sampler
out, time, lamb_list = selector.train(X, y, 100, method = 'ttts', bootstrap = True)

# Plot posterior probabilities
plot_posterior_mean(out, time, q,
                    title = 'Posterior probabilities over time',
                    save_name = 'figures/linear_tlasso_bag.png')

# Observe the penalty sizes
plt.plot(lamb_list)
plt.show()

# Run LASSO on the entire dataset
pooled = LassoCV(cv = 5, max_iter = 10000).fit(X, y)
print(pooled.alpha_)
print(pooled.coef_)
print(np.argwhere(abs(pooled.coef_) > 0.01).reshape(-1))

# Plot variable importance
temp = np.abs(pooled.coef_)
plt.scatter(range(temp.shape[0]), temp)
plt.scatter(range(5), temp[:5], color = 'red')
plt.grid(True)
plt.title(f"LASSO")
plt.xlabel("Variable")
plt.ylabel(f"Coefficient magnitude")
plt.savefig('figures/linear_lasso.png')
plt.close()

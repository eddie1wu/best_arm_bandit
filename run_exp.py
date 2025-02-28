from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

from ImportanceBandit import ImportanceBandit
from utils import *

n = 300
p = 1000
q = 5
# coefs = np.zeros((p, ))
# coefs[:q] = np.full((q, ), 10)
# X, y, true_coef = make_linreg(
#     n_samples = n,
#     n_features = p,
#     n_informative = q,
#     intercept = 10,
#     std_error = 5,
#     specify_coef = coefs,
#     return_coef = True
# )
#
# print(f"The true coefs are {true_coef[:q]}.")

X, y = make_friedman(
    n_samples = n,
    n_features = p,
    std_error = 1
)


print(f"X has shape {X.shape}.")
print(f"y has shape {y.shape}.")



# linreg = LinearRegression()
#
# # Create the bandit for variable importance
# alpha_prior = np.ones(p)
# beta_prior = np.ones(p)
# selector = ImportanceBandit(linreg, alpha_prior, beta_prior, max_features = 0.03)
#
# # Train the sampler
# truth = np.arange(5)
# num_epochs = None
# probabilities, time = selector.train(X, y, num_epochs, threshold = 0.1, method = 'ttts', bootstrap = True, max_iter = 500,
#                                      verbose = 10)
#
# # Plot posterior probabilities
# plot_posterior_mean(probabilities, time, q,
#                     title = 'Posterior probabilities over time',
#                     save_name = 'figures/best_m_lasso.png')
#
# print(probabilities[-1])



# Create the model
rf = RandomForestRegressor(
    n_estimators = 100,
    max_depth = 10,
    max_features = 1,
    n_jobs = -1
)

# Create the bandit for variable importance
alpha_prior = np.ones(p)
beta_prior = np.ones(p)
selector = ImportanceBandit(rf, alpha_prior, beta_prior, bootstrap = False, time_series = False)

# Train the sampler
num_epochs = 200

probabilities, time, r2_list = selector.train(
    X,
    y,
    num_epochs = num_epochs,
    method = 'ts',
    how_reward = 'absolute',
    threshold = 0.02,
    max_features = 0.2
)

# Plot posterior probabilities
plot_posterior_mean(probabilities, time, q,
                    title = 'Posterior probabilities over time',
                    save_name = 'figures/best_m_lasso.png')

print(r2_list)
print(len(r2_list))
print(time)

plt.scatter(range(time), r2_list)
plt.show()

print(probabilities[-1])






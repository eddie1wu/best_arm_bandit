from sklearn.ensemble import RandomForestRegressor

from ImportanceBandit import ImportanceBandit

from utils import *

n = 200
p = 200
q = 5

# Create the bandit for variable importance
alpha_prior = np.ones(p)
beta_prior = np.ones(p)

rf = RandomForestRegressor(
    n_estimators = 100,
    max_depth = 10,
    max_features = 1,
    n_jobs = -1
)

num_iter = 10

selector = ImportanceBandit(rf, alpha_prior, beta_prior, bootstrap = False, time_series = False)

probabilities = []
r2_list = []

for t in range(num_iter):

    X, y = make_friedman(
        n_samples = n,
        n_features = p,
        std_error = 1
    )

    probability, _, r2 = selector.train(
        X,
        y,
        num_epochs = 1,
        method = 'ttts',
        how_reward = 'absolute',
        threshold = 0.02,
        max_features = 0.5,
        verbose = 1,
        disable_progress = False
    )
    probabilities.append(probability[0])
    r2_list.append(r2[0])

# Plot posterior probabilities
plot_posterior_mean(probabilities, num_iter, q,
                    title = 'Posterior probabilities over time',
                    save_name = 'figures/best_m_lasso.png')

print(r2_list)
print(len(r2_list))
print(num_iter)

plt.scatter(range(num_iter), r2_list)
plt.show()

print(probabilities[-1])




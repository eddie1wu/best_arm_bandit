import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from BinaryBandit import *
from Sampler import *
from utils import *

# Define true parameters
params = np.array([
    0.3, 0.3, 0.3, 0.3, 0.4
])

# Define prior
alpha_prior = np.ones(params.shape)
beta_prior = np.ones(params.shape)

# Initialize bandit
bandit = BinaryBandit(params)

# Set the number of iterations in simulation
num_iter = 10

# Save results
results = {
    'uniform': [],
    'ts': [],
    'ttts': [],
    'ttps': [],
    'tempered_ts': [],
    'exploration_sampling': []
}

# Simulation
for i in tqdm(range(num_iter)):

    for method in results.keys():
        algo = Sampler(alpha_prior, beta_prior, bandit)
        time, _ = algo.time_to_convergence(getattr(algo, method), algo.arm_optimal_monte_carlo, 0.9)
        results[method].append(time)

# Plot
means = {method: np.mean(times) for method, times in results.items()}
plot_statistic(means,
               'Mean Time to Convergence by Algorithm',
               'figures/compare_mean.png')

stds = {method: np.std(times) for method, times in results.items()}
plot_statistic(stds,
               'Std of Time to Convergence by Algorithm',
               'figures/compare_std.png')

# methods = list(means.keys())
# mean_values = list(means.values())
#
# # Plot the bar chart
# plt.bar(methods, mean_values)
#
# plt.xticks(rotation = 10)
# plt.xlabel('Algorithm')
# plt.ylabel('Mean Time to Convergence (0.99)')
# plt.title('Mean Time to Convergence by Algorithm')
#
# plt.savefig('figures/compare_mean.png', dpi = 400)
# plt.close()

# Show the plot
# plt.show()

# # Plot
# stds = {method: np.std(times) for method, times in results.items()}
#
# # Extract methods and mean values for plotting
# methods = list(stds.keys())
# stds_values = list(stds.values())
#
# # Plot the bar chart
# plt.bar(methods, stds_values)
#
# plt.xticks(rotation = 10)
# plt.xlabel('Algorithm')
# plt.ylabel('Std of Time to Convergence (0.99)')
# plt.title('Std of Time to Convergence by Algorithm')
#
# plt.savefig('figures/compare_std.png', dpi = 400)
# plt.close()

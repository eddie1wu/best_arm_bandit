import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from BinaryBandit import BinaryBandit
from Sampler import Sampler
from utils import *

def compare_algo(params):
    """
    Compare the number of iterations needed for convergence for different sampling methods.
    """
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

if __name__ == '__main__':

    # Define true parameters
    params = np.array([
        0.2, 0.4, 0.6, 0.7, 0.8
    ])

    compare_algo(params)

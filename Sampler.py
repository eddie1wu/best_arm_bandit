import numpy as np
from scipy.stats import beta
from scipy.integrate import quad

class Sampler:
    def __init__(self, alpha_prior, beta_prior, bandit):

        self.bandit = bandit
        self.n_arms = bandit.n_arms
        self.best_arm = bandit.best_arm

        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior

        self.alpha_post = alpha_prior.copy()
        self.beta_post = beta_prior.copy()

    def arm_optimal_monte_carlo(self, arm, n_sim=50000):
        # Evaluate the probability that an arm is optimal by monte carlo
        temp = np.random.beta(self.alpha_post[np.newaxis, :], self.beta_post[np.newaxis, :], size=(n_sim, self.n_arms))
        p = np.sum(np.argmax(temp, axis=1) == arm) / n_sim

        return p

    def arm_optimal_integral(self, arm):
        # Evaluate the probability that an arm is optimal by numerical integration

        def integrand(x):
            # PDF of x
            f_x = beta.pdf(x, self.alpha_post[arm], self.beta_post[arm])

            # CDFs of the other arms
            product_cdf = 1
            for i in range(self.n_arms):
                if i == arm:
                    continue
                product_cdf *= beta.cdf(x, self.alpha_post[i], self.beta_post[i])

            return f_x * product_cdf

        return quad(integrand, 0, 1)[0]

    def uniform(self):
        # Samples an arm with random uniform
        arm = np.random.randint(self.n_arms)

        return arm

    def ts(self):
        # Thompson sampling
        theta_array = np.random.beta(self.alpha_post, self.beta_post)
        arm = np.argmax(theta_array)

        return arm

    def ttts(self, beta=0.5):
        # Top two Thompson sampling
        arm = self.ts()

        if np.random.rand() < beta:
            return arm
        else:
            second_arm = arm
            while second_arm == arm:
                second_arm = self.ts()
            return second_arm

    def ttps(self, beta=0.5):
        n_sim = 50000
        temp = np.random.beta(self.alpha_post[np.newaxis, :], self.beta_post[np.newaxis, :], size=(n_sim, self.n_arms))

        def func(x):
            return np.sum(np.argmax(temp, axis=1) == x) / n_sim

        array = np.arange(self.n_arms).reshape(-1, 1)
        probability_vector = np.apply_along_axis(func, axis=1, arr=array)
        probability_vector = probability_vector / np.sum(probability_vector)

        order = np.argsort(probability_vector)

        if np.random.rand() < beta:
            idx = -1
        else:
            idx = -2

        return order[idx]

    def tempered_ts(self, beta=0.8):
        # Tempered Thompson sampling in Kasy et al 2023

        if np.random.rand() < beta:
            return self.ts()
        else:
            return self.uniform()

    def exploration_sampling(self):
        # Kasy and Sautmann 2021 Econometrica
        n_sim = 50000
        temp = np.random.beta(self.alpha_post[np.newaxis, :], self.beta_post[np.newaxis, :], size=(n_sim, self.n_arms))

        def func(x):
            return np.sum(np.argmax(temp, axis=1) == x) / n_sim

        array = np.arange(self.n_arms).reshape(-1, 1)
        probability_vector = np.apply_along_axis(func, axis=1, arr=array)

        # Use the transformed shares
        probability_vector = probability_vector * (1 - probability_vector)
        probability_vector = probability_vector / np.sum(probability_vector)

        arm = np.random.choice(np.arange(self.n_arms), p=probability_vector)

        return arm

    def update(self, x, arm):

        self.alpha_post[arm] += x
        self.beta_post[arm] += (1 - x)

    def time_to_convergence(self, policy, arm_optimal_method, threshold):

        p = 0
        time = 0

        while p < threshold:
            arm = policy()
            x = self.bandit.draw_sample(arm)
            self.update(x, arm)

            time += 1
            p = arm_optimal_method(self.best_arm)

        return time, p


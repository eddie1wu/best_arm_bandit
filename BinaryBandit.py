import numpy as np

class BinaryBandit:
    def __init__(self, params):

        self.true_params = params
        self.n_arms = params.shape[0]
        self.best_arm = np.argmax(params)

    def draw_sample(self, k):

        return np.random.binomial(n = 1, p = self.true_params[k])
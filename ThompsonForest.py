import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from tqdm import tqdm

class ThompsonLasso:
    def __init__(self, alpha_prior, beta_prior, C, threshold = 0.1):

        self.alpha = alpha_prior
        self.beta = beta_prior
        self.C_tilde = np.log(1/C) / np.log( (C+1)/C )
        self.threshold = threshold # threshold for determining local reward based on RF permutation importance

    def choose(self):
        """
        Choose the variables to be assessed
        """
        theta = np.random.beta(self.alpha, self.beta)
        idx = np.argwhere(theta >= self.C_tilde).reshape(-1)

        return idx

    def toptwo_choose(self):
        """
        Use top two TS to choose the variables to be assessed
        """
        idx = self.choose()
        beta = np.random.rand()
        if beta < 0.5:
            return idx
        else:
            idx2 = self.choose()
            while np.array_equal(idx, idx2):
                idx2 = self.choose()
            return idx2

    def reward(self, idx, X, Y):
        """
        Compute local reward for each chosen variable
        """
        xx = X[:, idx]

        X_train, X_test, y_train, y_test = train_test_split(xx, y, test_size = 0.2)

        rf = RandomForestRegressor(n_estimators = 200,
                                    max_depth = 6,
                                    min_samples_split = 2,
                                    min_samples_leaf = 1,
                                    max_features = 1.0,
                                    # oob_score = True,
                                    n_jobs = -1)
        rf.fit(X_train, y_train)

        baseline = rf.score(X_test, y_test)

        result = permutation_importance(
            rf, X_test, y_test, n_repeats = 10, n_jobs = -1
        )

        importance_score = np.clip(result.importances_mean / baseline, 0, 1)

        # reward = np.random.binomial(1, importance_score).astype(bool)
        reward = importance_score >= self.threshold

        good_arms = idx[reward]
        bad_arms = idx[~reward]

        return good_arms, bad_arms

    def update(self, good_arms, bad_arms):
        """
        Update alpha and beta
        """
        self.alpha[good_arms] += 1
        self.beta[bad_arms] += 1

        return self.alpha / (self.alpha + self.beta)

    def train(self, X, Y, num_epochs, method = 'ttts', threshold = None):
        """
        Train the model
        """
        if threshold:
            self.threshold = threshold

        time = 0
        out = []

        for i in tqdm(range(num_epochs)):

            time += 1

            if method == 'ttts':
                idx = self.toptwo_choose()
            elif method == 'ts':
                idx = self.choose()

            good_arms, bad_arms = self.reward(idx, X, Y)
            pi = self.update(good_arms, bad_arms)

            out.append(pi)

        return out, time


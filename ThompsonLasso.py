import numpy as np
from sklearn.linear_model import LassoCV, Lasso
from tqdm import tqdm

# Top two thompson sampling, with bagging.
class ThompsonLasso:
    def __init__(self, alpha_prior, beta_prior, C, threshold = 0.1):
        """
        Iterative LASSO variable selection with subsets chosen by Thompson sampling
        """
        self.alpha = alpha_prior
        self.beta = beta_prior
        self.C_tilde = np.log(1/C) / np.log( (C+1)/C )
        self.threshold = threshold # threshold for determining local reward based on lasso coefs

    def choose(self):
        """
        Choose the variables to be assessed
        """
        theta = np.random.beta(self.alpha, self.beta)
        idx = np.argwhere(theta >= self.C_tilde).reshape(-1)

        return idx

    def toptwo_choose(self):
        """
        Use top two Thompson sampling to choose the variables to be assessed
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

    def random_choose(self, k):
        """
        Sample k arms uniformly
        """
        idx = np.random.choice(np.arange(self.alpha.shape[0]), k)
        return idx

    def reward(self, idx, X, Y, bootstrap = False, lamb = None):
        """
        Compute local reward for each chosen variable
        """
        # Bootstrap
        outcome = 0

        if bootstrap:
            n_iter, inclusion_count = 5, 5
        else:
            n_iter, inclusion_count = 1, 1

        for i in range(n_iter):
            resample = np.random.randint(X.shape[0], size = X.shape[0])
            xx = X[np.ix_(resample, idx)]
            yy = Y[resample]

            # Choose alpha
            if lamb:
                opt_lamb = lamb
            else:
                # Run Lasso with optimal alpha via cross validation
                lasso = LassoCV(cv = 5, max_iter = 10000, random_state = 0).fit(xx, yy)

                # Make sure that Lasso is not too close to 0
                opt_lamb = max(lasso.alpha_*2, 0.025)

            lasso = Lasso(alpha = opt_lamb, max_iter = 10000, random_state = 0).fit(xx, yy)

            out = np.abs(lasso.coef_) > self.threshold
            outcome += out.astype(int)

        outcome = outcome >= inclusion_count
        good_arms = idx[outcome]
        bad_arms = idx[~outcome]

        return good_arms, bad_arms, opt_lamb

    def update(self, good_arms, bad_arms):
        """
        Update alpha and beta
        """
        self.alpha[good_arms] += 1
        self.beta[bad_arms] += 1

        return self.alpha / (self.alpha + self.beta)

    def train(self, X, Y, num_epochs, method = 'ts', bootstrap = False, threshold = None, lamb = None):
        """
        Train the model
        """
        if threshold:
            self.threshold = threshold

        time = 0
        out = []
        lamb_list = []

        for i in tqdm(range(num_epochs)):

            time += 1

            if method == 'ttts':
                idx = self.toptwo_choose()
            elif method == 'ts':
                idx = self.choose()
            elif method == 'random':
                idx = self.random_choose(50)
            good_arms, bad_arms, opt_lamb = self.reward(idx, X, Y, bootstrap, lamb)
            pi = self.update(good_arms, bad_arms)

            out.append(pi)
            lamb_list.append(opt_lamb)

        return out, time, lamb_list

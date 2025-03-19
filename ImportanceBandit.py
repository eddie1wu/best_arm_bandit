import numpy as np

from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from tqdm import tqdm

class ImportanceBandit:
    """
    Bandit for determining variable importance in a model

    Attributes:
        model: a model object
        alpha_prior: an array of alpha priors with length equal to the number of bandit arms.
        beta_prior: an array of beta priors.
        max_features: a float indicating the maximum fraction of features to be included at each time step.
        bootstrap: a bool for whether to get a bootstrap dataset at every iteration.
        time_series: a bool for whether to maintain temporal ordering.
    """

    def __init__(self,
                 model,
                 alpha_prior,
                 beta_prior,
                 max_features=1,
                 bootstrap = True,
                 time_series = False):

        self.model = model
        self.alpha = alpha_prior
        self.beta = beta_prior
        self.max_num_features = int(max_features * len(self.alpha))
        self.bootstrap = bootstrap
        self.time_series = time_series

    def limit_features(self, idx):

        if len(idx) > self.max_num_features:
            idx = np.random.choice(idx, size = self.max_num_features, replace = False)

        return idx

    def choose(self):
        """
        Thompson sampling for combinatorial bandit, with MPM oracle.
        """

        theta = np.random.beta(self.alpha, self.beta)
        idx = np.argwhere(theta >= 0.5).reshape(-1)

        return self.limit_features(idx)

    def top_two_choose(self):
        """
        With probability 0.5, obtain subset 1 and play it.
        With probability 0.5, sample subset 2 which is different from subset 1,
        and play the union minus intersection of the two subsets.
        """
        idx = self.choose()
        if np.random.rand() < 0.5:
            return idx

        else:
            idx2 = self.choose()
            while np.array_equal(idx, idx2): # Re-sample if same as subset1
                idx2 = self.choose()
            idx3 = np.concatenate( (np.setdiff1d(idx, idx2), np.setdiff1d(idx2, idx)) )

            return self.limit_features(idx3)

    def reward(self, idx):
        """
        Obtain local rewards for each of the selected variables
        """
        # Select columns
        xx = self.X[:, idx]

        # Train test split. If time series data, do not shuffle.
        X_train, X_test, y_train, y_test = train_test_split(xx, self.y, test_size=0.2, shuffle = not self.time_series)

        # Bootstrap if True
        if self.bootstrap:
            sample = np.random.randint(len(X_train), size = len(X_train))
        else:
            sample = range(len(X_train))
        X_train = X_train[sample]
        y_train = y_train[sample]

        # Obtain local reward
        self.model.fit(X_train, y_train)
        out_sample_r2 = self.model.score(X_test, y_test)
        result = permutation_importance(
            self.model,
            X_test,
            y_test,
            n_repeats=10,
            n_jobs=-1
        )
        importance_score = result.importances_mean
        if self.how_reward == 'relative':
            importance_score = importance_score / abs(out_sample_r2)
        reward = importance_score >= self.threshold

        # Assign reward arms
        good_arms = idx[reward]
        bad_arms = idx[~reward]

        return good_arms, bad_arms, out_sample_r2

    def update(self, good_arms, bad_arms):
        """
        Update alpha and beta, return the expected value of the beta distribution
        """
        self.alpha[good_arms] += 1
        self.beta[bad_arms] += 1

        return self.alpha / (self.alpha + self.beta)

    def eval_convergence(self, features_informative = None, probabilities = None):
        """
        If true features are known, convergence is evaluated based on true features.
        Otherwise, convergence is evaluated based on the stability of posterior inclusion probabilities.
        """
        if features_informative is not None:
            # Given true informative features, convergence is when
            # the true features have the highest posterior probabilities
            # few other features have posterior probabilities greater than 0.5.
            k = len(features_informative)
            curr_mu = self.alpha / (self.alpha + self.beta)
            curr_best = np.argsort(curr_mu)[-k:]

            noise_arr = np.delete(curr_mu, features_informative) # Delete the probabilities on the features_informative indices

            condition1 = np.array_equal( np.sort(features_informative), np.sort(curr_best) ) & np.all(curr_mu[:k] > 0.5) # All top features more than 50% prob.
            condition2 = np.sum(noise_arr >= 0.5) == 0  # previously was k/2

            return condition1 and condition2

        elif probabilities is not None:

            if len(probabilities) > 50:
                arr = np.array(probabilities[-50:])
                k = np.sum(arr[-1] > 0.5)
                top_arms = [np.argsort(row)[-k:] for row in arr]

                return all(np.array_equal(arms, top_arms[-1]) for arms in top_arms)

            else:
                return False

    def train(
            self,
            X,
            y,
            num_epochs = None,
            method='ts',
            how_reward = 'relative',
            threshold = 0.1,
            features_informative = None,
            max_iter = 5000,
            verbose = 0,
            disable_progress = False
    ):
        self.X = X
        self.y = y
        self.how_reward = how_reward
        self.threshold = threshold

        if method == 'ts':
            choose = self.choose
        elif method == 'ttts':
            choose = self.top_two_choose

        time_step = 0
        probabilities = []
        r2_list = []
        converged = False

        with tqdm(total = max_iter, disable = disable_progress) as pbar:
            while not converged:

                time_step += 1

                idx = choose()
                good_arms, bad_arms, r2 = self.reward(idx)

                r2_list.append(r2)

                curr_probabilities = self.update(good_arms, bad_arms)

                probabilities.append(curr_probabilities)

                if verbose > 0 and time_step % verbose == 0:
                    print(f"The arms chosen are {idx}")

                pbar.update(1)

                if num_epochs is not None and time_step == num_epochs:
                    break

                if time_step == max_iter:
                    print(f"Max iter {max_iter} reached. No convergence.")
                    break

                if num_epochs is None:
                    if features_informative is not None:
                        converged = self.eval_convergence(features_informative = features_informative)
                    else:
                        converged = self.eval_convergence(probabilities = probabilities)
                        # if converged:
                        #     time_step -= 49
                        #     probabilities = probabilities[:-49]

        return probabilities, time_step, r2_list

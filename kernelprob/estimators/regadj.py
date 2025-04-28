import numpy as np
import scipy
import sklearn.ensemble

from ..base_estimator import BaseEstimator
from ..utils.p_generator import get_p
from ..exact.treeprob import tree_prob

class NullModel:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.zeros(X.shape[0])

def get_fit(X_flat, y_flat, weighting, regression_adj):
    n = X_flat.shape[1]
    if regression_adj is False:
        reg_model = NullModel()
        phi_method = lambda reg_model : np.zeros(n)
    elif regression_adj == "linear":
        reg_model = sklearn.linear_model.LinearRegression()
        phi_method = lambda reg_model: reg_model.coef_
    elif regression_adj == "tree":
        reg_model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, max_features=n)
        phi_method = lambda reg_model: tree_prob(np.zeros((1,n)), np.ones((1,n)), reg_model, weighting)[0].squeeze()
    else:
        raise ValueError("regression_adjustment must be False, 'linear', or 'tree'")

    reg_model.fit(X_flat, y_flat)
    phi = phi_method(reg_model)

    return reg_model, phi

class GeneralMonteCarloEstimator(BaseEstimator):
    def __init__(self, model, baseline: np.ndarray, weighting: str, regression_adjust=False, use_median=False):
        super().__init__(model, baseline, weighting)

        self.n = self.baseline.shape[1] if self.baseline.ndim > 1 else self.baseline.size
        self.p = get_p(self.n, weighting)
        self.regression_adjust = regression_adjust
        self.use_median = use_median
        
        # pair sample if p is symmetric
        self.pair_sampling = np.allclose(self.p, self.p[::-1])
        self.per_S = 2 if not self.pair_sampling else 4

        #self.sample_prob = np.ones(self.n)
        self.sample_prob = self.p * scipy.special.binom(self.n-1, np.arange(self.n))
        self.sample_prob /= np.sum(self.sample_prob)
        self.rng = np.random.default_rng()
    
    def add_sample(self, i, mean_idx, S_idx, indices):
        size = len(indices)
        weight = self.p[size] * scipy.special.binom(self.n-1, size) / self.sample_prob[size]

        # Add S
        self.X[i, mean_idx, S_idx, indices] = 1
        self.sign[i, mean_idx, S_idx] = -1
        self.weights[i, mean_idx, S_idx] = weight

        # Add S cup i
        indices_with_i = np.append(indices, i)
        self.X[i, mean_idx, S_idx+1, indices_with_i] = 1
        self.sign[i, mean_idx, S_idx+1] = 1
        self.weights[i, mean_idx, S_idx+1] = weight
    
    def sample_with_replacement(self):
        for i in range(self.n):
            except_i = np.delete(np.arange(self.n), i)
            for mean_idx in range(self.num_means):
                sampled_sizes = self.rng.choice(self.n, self.nue_avg, p=self.sample_prob)
                
                for S_idx, size in enumerate(sampled_sizes):
                    indices = self.rng.choice(except_i, size=size, replace=False)

                    self.add_sample(i, mean_idx, self.per_S * S_idx, indices)
                    
                    if self.pair_sampling:
                        indices_complement = np.delete(np.arange(self.n), np.append(indices, i))
                        self.add_sample(i, mean_idx, self.per_S * S_idx + 2, indices_complement)

    def explain(self, explicand: np.ndarray, num_samples: int, return_direct=False) -> np.ndarray:
        samples_per_i = num_samples // self.n
        self.S_per_i = samples_per_i // self.per_S
        if self.use_median:
            self.num_means = min(int(np.log(self.n)), self.S_per_i)
        else:
            self.num_means = 1
        self.nue_avg = self.S_per_i // self.num_means

        self.X = np.zeros((self.n, self.num_means, self.nue_avg*self.per_S, self.n), dtype=float)
        self.y = np.zeros((self.n, self.num_means, self.nue_avg*self.per_S), dtype=float)
        self.sign = np.ones((self.n, self.num_means, self.nue_avg*self.per_S), dtype=float)
        self.weights = np.ones((self.n, self.num_means, self.nue_avg*self.per_S), dtype=float)

        self.sample_with_replacement()
        
        X_flat = self.X.reshape(-1, self.n)
        model_input = self.baseline * (1 - X_flat) + explicand * X_flat
        y_flat = self.model.predict(model_input)
        y = y_flat.reshape(self.n, self.num_means, self.nue_avg*self.per_S)

        reg_model, reg_phi = get_fit(X_flat, y_flat, self.weighting, self.regression_adjust)
        reg_pred = reg_model.predict(X_flat).reshape(self.n, self.num_means, self.nue_avg*self.per_S)

        if return_direct: return reg_model
 
        phi = np.zeros(self.n)

        for i in range(self.n):
            means = []
            for mean_idx in range(self.num_means):
                means.append(((
                    y[i, mean_idx, :] * self.sign[i, mean_idx, :] 
                    - reg_pred[i, mean_idx, :] * self.sign[i, mean_idx, :]
                    + reg_phi[i]
                ) * self.weights[i, mean_idx, :]
                ).mean()
                )
            phi[i] = np.median(means)

        return phi

class WeightedMonteCarloEstimator(GeneralMonteCarloEstimator):
    def __init__(self, model, baseline: np.ndarray, weighting: str):
        super().__init__(model, baseline, weighting, regression_adjust=False, use_median=False)

class TreeAdjustedMC(GeneralMonteCarloEstimator):
    def __init__(self, model, baseline: np.ndarray, weighting: str):
        super().__init__(model, baseline, weighting, regression_adjust="tree")

class LinearAdjustedMC(GeneralMonteCarloEstimator):
    def __init__(self, model, baseline: np.ndarray, weighting: str):
        super().__init__(model, baseline, weighting, regression_adjust="linear")

class PermutationRegression(BaseEstimator):
    def __init__(self, model, baseline: np.ndarray, weighting: str):
        super().__init__(model, baseline, weighting)
        self.n = self.baseline.shape[1] if self.baseline.ndim > 1 else self.baseline.size
        self.p = get_p(self.n, weighting)        
        self.rng = np.random.default_rng()
    
    def explain(self, explicand: np.ndarray, num_samples: int) -> np.ndarray:
        num_permutations = (num_samples - 2) // (self.n-1)
        phi = np.zeros(self.n)
        pred_baseline = self.model.predict(self.baseline)
        pred_explicand = self.model.predict(explicand)

        X = np.zeros((num_permutations,self.n-1, self.n))
        permutations = np.zeros((num_permutations, self.n))
        for perm_idx in range(num_permutations):
            if perm_idx % 2 == 0: # Sample a random permutation
                permutation = self.rng.permutation(self.n)
            else: # Reverse the previous permutation
                permutation = permutation[::-1]
            permutations[perm_idx, :] = permutation

        for perm_idx in range(num_permutations):
            permutation = permutations[perm_idx].astype(int)
            for i in range(self.n-1):
                X[perm_idx, i, permutation[:i+1]] = 1        

        X_flat = X.reshape(-1, self.n)
        model_input = self.baseline * (1 - X_flat) + explicand * X_flat
        y_flat = self.model.predict(model_input)
        y = y_flat.reshape(num_permutations, self.n-1)

        reg_model, reg_phi = get_fit(X_flat, y_flat, self.weighting, "tree")
        reg_pred = reg_model.predict(X_flat).reshape(num_permutations, self.n-1)
        reg_baseline = reg_model.predict(np.zeros_like(self.baseline))
        reg_explicand = reg_model.predict(np.ones_like(explicand))

        for perm_idx in range(num_permutations):
            permutation = permutations[perm_idx].astype(int)
            for i in range(self.n):
                if i == 0:
                    phi[permutation[i]] += (
                        y[perm_idx, i] - pred_baseline
                        - reg_pred[perm_idx, i] + reg_baseline
                        + reg_phi[permutation[i]]
                    ) / num_permutations
                elif 0 < i and i < self.n-1:
                    phi[permutation[i]] += (
                        y[perm_idx, i] - y[perm_idx, i-1]
                        - reg_pred[perm_idx, i] + reg_pred[perm_idx, i-1]
                        + reg_phi[permutation[i]]
                    ) / num_permutations
                elif i == self.n-1:
                    phi[permutation[i]] += (
                        pred_explicand - y[perm_idx, i-1]
                        - reg_explicand + reg_pred[perm_idx, i-1]
                        + reg_phi[permutation[i]]
                    ) / num_permutations

        return phi

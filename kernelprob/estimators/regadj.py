import numpy as np
import scipy
import sklearn.ensemble

from ..base_estimator import BaseEstimator
from ..utils.p_generator import get_p
from ..exact.treeprob import tree_prob

class RegressionAdjustmentEstimator(BaseEstimator):
    def __init__(self, model, baseline: np.ndarray, weighting: str, model_type='random_forest'):
        super().__init__(model, baseline, weighting)
        self.model_type = model_type

        self.n = self.baseline.shape[1] if self.baseline.ndim > 1 else self.baseline.size
        self.p = get_p(self.n, weighting)
        
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

    def explain(self, explicand: np.ndarray, num_samples: int) -> np.ndarray:
        samples_per_i = num_samples // self.n
        self.S_per_i = samples_per_i // self.per_S
        self.num_means = min(int(np.log(self.n)), self.S_per_i)
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
        
        # Fit on all data
        if self.model_type == 'linear':
            reg_model = sklearn.linear_model.LinearRegression()
            reg_model.fit(model_input, y_flat)
            reg_model_phi = reg_model.coef_
        
        elif self.model_type == 'exact_linear':
            try:
                true_values = tree_prob(self.baseline, explicand, self.model, self.weighting)[0][0]
            except:
                true_values = tree_prob(self.baseline, explicand, self.model.model, self.weighting)[0][0]
            
            reg_model = sklearn.linear_model.LinearRegression()
            reg_model.coef_ = true_values
            reg_model.intercept_ = self.model.predict(self.baseline)[0]
            reg_model_phi = true_values

        elif self.model_type == 'random_forest':
            reg_model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, max_features=self.n)
            reg_model.fit(model_input, y_flat)
            reg_model_phi = tree_prob(self.baseline, explicand, reg_model, self.weighting)[0].squeeze()
        
        reg_pred_flat = reg_model.predict(model_input)
        reg_pred = reg_pred_flat.reshape(self.n, self.num_means, self.nue_avg*self.per_S)

        

        phi = np.zeros(self.n)

        for i in range(self.n):
            means = []
            for mean_idx in range(self.num_means):
                means.append(((
                    y[i, mean_idx, :] * self.sign[i, mean_idx, :] 
                    - reg_pred[i, mean_idx, :] * self.sign[i, mean_idx, :]
                    + reg_model_phi[i]
                ) * self.weights[i, mean_idx, :]
                ).mean()
                )
            phi[i] = np.median(means)

        return phi

class PermutationRegression(BaseEstimator):
    def __init__(self, model, baseline: np.ndarray, weighting: str):
        super().__init__(model, baseline, weighting)
        self.n = self.baseline.shape[1] if self.baseline.ndim > 1 else self.baseline.size
        self.p = get_p(self.n, weighting)        
        self.rng = np.random.default_rng()
    
    def explain(self, explicand: np.ndarray, num_samples: int) -> np.ndarray:
        num_permutations = (num_samples - 2)// (self.n-1)
        phi = np.zeros(self.n)
        pred_baseline = self.model.predict(self.baseline)
        pred_explicand = self.model.predict(explicand)

        X = np.zeros((num_permutations*num_samples, self.n))
        for perm_idx in range(num_permutations):
            # Set seed to perm_idx
            self.rng = np.random.default_rng(perm_idx)
            permutation = self.rng.permutation(self.n)
            for i in range(1, self.n):
                X[perm_idx*self.n + i, permutation[:i]] = 1
        y = self.model.predict(self.baseline * (1 - X) + explicand * X)
        reg_model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, max_features=self.n)
        reg_model.fit(X, y)
        reg_model_phi = tree_prob(self.baseline, explicand, reg_model, self.weighting)[0].squeeze()
        reg_baseline = reg_model.predict(self.baseline)
        reg_explicand = reg_model.predict(explicand)
        reg_pred = reg_model.predict(X)

        for perm_idx in range(num_permutations):
            self.rng = np.random.default_rng(perm_idx)
            permutation = self.rng.permutation(self.n)
            for i in range(1, self.n+1):
                if i == 1:
                    phi[permutation[i-1]] += (
                        y[perm_idx*self.n + i] - pred_baseline
                        #- reg_pred[perm_idx*self.n + i] + reg_baseline
                        #+ reg_model_phi[permutation[i-1]]
                    ) / num_permutations
                elif i == self.n:
                    phi[permutation[i-1]] += (
                        pred_explicand - y[perm_idx*self.n + i]
                        #- reg_explicand + reg_pred[perm_idx*self.n + i]
                        #+ reg_model_phi[permutation[i-1]]
                    ) / num_permutations
                else:
                    phi[permutation[i-1]] += (
                        y[perm_idx*self.n + i] - y[perm_idx*self.n + i-1]
                        #- reg_pred[perm_idx*self.n + i] + reg_pred[perm_idx*self.n + i-1]
                        #+ reg_model_phi[permutation[i-1]]
                    ) / num_permutations
        return phi

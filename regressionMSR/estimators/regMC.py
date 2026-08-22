"""
Updated in accordance with commit ac4c5ac
"""

import numpy as np
import scipy
import sklearn.ensemble
import pandas as pd

import sklearn.ensemble
import sklearn.linear_model
from ..exact.treeshap import tree_shap
from ..exact.treeprob import tree_prob
from ..exact.enumeration import enumeration_prob

from ..base_estimator import BaseEstimator
from ..utils.p_generator import get_p, is_symmetric
from .est_utils import combination_generator
import xgboost

class NullModel:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.zeros(X.shape[0])

class LinearModel:
    def __init__(self, phi):
        self.coef_ = phi
    
    def fit(self, X, y):
        pass

    def predict(self, X):
        return X.values @ self.coef_
    
def leverage_shap(X, y, cache):
    prob_sampled, v1, v0, p = cache['prob_sampled'], cache['v1'], cache['v0'], cache['p']
    n = X.shape[1]
    sizes = X.sum(axis=1).astype(int)
    Sv = y - v0
    sum_weighting = - p[sizes] * (n - sizes) + p[sizes-1] * sizes
    sum_phi = prob_sampled * sum_weighting @ Sv
    sum_phi = sum_phi + (v1 - v0) * p[-1] * n

    Sb = Sv - sizes * sum_phi / n
    P = np.eye(n) - 1/n
    reg_weighting = 1/prob_sampled * (p[sizes-1] + p[sizes])
    PZSSb = P @ X.T @ np.diag(reg_weighting) @ Sb
    PZSSZP = P @ X.T @ np.diag(reg_weighting) @ X @ P
    PZZSZP_inv_PZSSb = np.linalg.lstsq(PZSSZP, PZSSb, rcond=None)[0]

    reg_phi = PZZSZP_inv_PZSSb + sum_phi / n
    return LinearModel(reg_phi), reg_phi    

def kernel_banzhaf(X, y):
    reg_phi = np.linalg.lstsq(X - 1/2, y, rcond=None)[0]
    return LinearModel(reg_phi), reg_phi

def get_fit(X_flat, y_flat, weighting, regression_adj, cache=None):
    n = X_flat.shape[1]
    if regression_adj is False:
        reg_model = NullModel()
        phi_method = lambda reg_model : np.zeros(n)
    elif regression_adj == "linear" and weighting in ['shapley', 'beta_shapley_1_1']:
        return leverage_shap(X_flat.values, y_flat, cache)
    elif regression_adj == "linear" and weighting in ['banzhaf', 'weighted_banzhaf_0.5']:
        return kernel_banzhaf(X_flat.values, y_flat)
    elif regression_adj == "linear":
        reg_model = sklearn.linear_model.LinearRegression()
        phi_method = lambda reg_model: reg_model.coef_
    elif regression_adj == "tree":
        reg_model = xgboost.XGBRegressor()
        if weighting in ['shapley', 'beta_shapley_1_1']:
            phi_method = lambda reg_model: tree_shap(np.zeros((1,n)), np.ones((1,n)), reg_model)[0].squeeze()
        else:
            phi_method = lambda reg_model: tree_prob(np.zeros((1,n)), np.ones((1,n)), reg_model, weighting)[0].squeeze()
    else:
        raise ValueError("regression_adjustment must be False, 'linear', or 'tree'")

    reg_model.fit(X_flat, y_flat)
    phi = phi_method(reg_model)

    return reg_model, phi


class UniversalMC(BaseEstimator):
    def __init__(self, model, baseline: np.ndarray,
                 weighting: str, regression_adjust=False, use_median=False, seed=None):
        super().__init__(model, baseline, weighting)

        self.n = (self.baseline.shape[1] if self.baseline.ndim > 1 else self.baseline.size)
        self.p = get_p(self.n, weighting)
        if len(self.p) < self.n + 1:
            self.p = np.append(self.p, 0.0)
        self.regression_adjust = regression_adjust
        self.use_median        = use_median

        # NOTE:  we never need |S| = n here -> use p[:-1]
        self.sample_prob = self.p[:-1] * scipy.special.binom(
            self.n - 1, np.arange(self.n)
        )
        self.sample_prob /= self.sample_prob.sum()

        self.pair_sampling = is_symmetric(self.p)
        self.per_S         = 2 if not self.pair_sampling else 4
        self.rng           = np.random.default_rng(seed)  # seed=None: fresh entropy, not replayable
    
    def add_sample(self, i, mean_idx, S_idx, indices):
        size = len(indices)
        # weight = self.p[size] * scipy.special.binom(self.n-1, size) / self.sample_prob[size]
        weight = (self.p[size] * scipy.special.binom(self.n-1, size)
                  / self.sample_prob[size])
        # actual sampling probability of this row
        prob   = self.sample_prob[size] / scipy.special.binom(self.n-1, size)

        # Add S
        self.X[i, mean_idx, S_idx, indices] = 1
        self.sign[i, mean_idx, S_idx] = -1
        self.weights[i, mean_idx, S_idx] = weight
        self.prob_sampled[i, mean_idx, S_idx] = prob

        # Add S cup i
        indices_with_i = np.append(indices, i)
        self.X[i, mean_idx, S_idx+1, indices_with_i] = 1
        self.sign[i, mean_idx, S_idx+1] = 1
        self.weights[i, mean_idx, S_idx+1] = weight
        self.prob_sampled[i, mean_idx, S_idx+1] = prob
    
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
        
        self.prob_sampled = np.zeros_like(self.weights)

        self.sample_with_replacement()
        
        # X_flat = self.X.reshape(-1, self.n)
        # model_input = self.baseline * (1 - X_flat) + explicand * X_flat
        # y_flat = self.model.predict(model_input)

        # NOTE: Revised to use the new X, y, sign, weights, and prob_sampled
        X_flat = self.X.reshape(-1, self.n)
        model_input = self.baseline * (1 - X_flat) + explicand * X_flat
        y_flat     = self.model.predict(model_input)

        # Build cache for leverage-SHAP when needed
        cache = None
        if (self.regression_adjust == "linear"
                and self.weighting in ["shapley", "beta_shapley_1_1"]):
            v1 = float(np.asarray(self.model.predict(explicand)).reshape(-1)[0])
            v0 = float(np.asarray(self.model.predict(self.baseline)).reshape(-1)[0])
            cache = {
                "prob_sampled": self.prob_sampled.reshape(-1),
                "v1": v1,
                "v0": v0,
                "p": self.p
            }
        X_flat_df = pd.DataFrame(X_flat, columns=[f"feature_{j}"
                                                  for j in range(self.n)])

        y = y_flat.reshape(self.n, self.num_means, self.nue_avg*self.per_S)

        # reg_model, reg_phi = get_fit(X_flat, y_flat, self.weighting, self.regression_adjust)
        # reg_pred = reg_model.predict(X_flat).reshape(self.n, self.num_means, self.nue_avg*self.per_S)
        reg_model, reg_phi = get_fit(
            X_flat_df, y_flat, self.weighting, self.regression_adjust, cache
        )
        reg_pred = reg_model.predict(X_flat_df).reshape(
             self.n, self.num_means, self.nue_avg*self.per_S)

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

        if (self.regression_adjust == "linear"
                and self.weighting in ["shapley", "beta_shapley_1_1"]):
            phi += (v1 - reg_phi.sum()) / self.n - v0 / self.n

        return phi

class WeightedMC(UniversalMC):
    def __init__(self, model, baseline: np.ndarray, weighting: str, seed=None):
        super().__init__(model, baseline, weighting, regression_adjust=False, use_median=False, seed=seed)

class TreeMC(UniversalMC):
    def __init__(self, model, baseline: np.ndarray, weighting: str, seed=None):
        super().__init__(model, baseline, weighting, regression_adjust="tree", seed=seed)

class LinearMC(UniversalMC):
    def __init__(self, model, baseline: np.ndarray, weighting: str, seed=None):
        super().__init__(model, baseline, weighting, regression_adjust="linear", seed=seed)
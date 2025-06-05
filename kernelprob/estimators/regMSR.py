import numpy as np
import scipy
from scipy.special import comb
import scipy.special
import pandas as pd

import sklearn.ensemble
import sklearn.linear_model
from ..exact.treeshap import tree_shap
from ..exact.treeprob import tree_prob
from ..exact.enumeration import enumeration_prob
from ..estimators.reg import UniversalRegression

from ..base_estimator import BaseEstimator
from ..utils.p_generator import get_p
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

class UniversalMSR(BaseEstimator):
    def __init__(self, model, baseline, weighting, reg_model_class=False, return_direct=False):
        super().__init__(model, baseline, weighting)
        self.n = self.baseline.shape[1]
        self.p = get_p(self.n, weighting)
        self.model = model
        self.baseline = baseline
        self.gen = np.random.Generator(np.random.PCG64())
        self.reg_model_class = reg_model_class
        self.valid_sizes = np.arange(self.n+1)
        self.pair_sampling = np.allclose(self.p, self.p[::-1])
        self.v1 = self.v0 = None        
        self.adjust = 1 if (self.reg_model_class == 'linear' and self.weighting in ['shapley', 'beta_shapley_1_1']) else 0

        self.sample_dist = []
        for size in range(self.n+1):
            prob = 0
            if size > 0:
                prob += self.p[size-1]**2 * size
            if size < self.n:
                prob += self.p[size]**2 * (self.n - size)
            self.sample_dist.append( np.sqrt(prob) )
        

        if reg_model_class == 'linear':
            # Leverage SHAP
            if weighting in ['shapley', 'beta_shapley_1_1']:
                s = np.arange(1, self.n)
                self.sample_dist = (self.p[s] + self.p[s-1]) * s * (self.n- s)
                self.valid_sizes = s
            # Kernel Banzhaf
            if weighting in ['banzhaf', 'weighted_banzhaf_0.5']:
                self.sample_dist = np.ones_like(self.sample_dist)

        self.sample_dist = np.array(self.sample_dist) / np.sum(self.sample_dist)
        self.return_direct = return_direct
    
    def add_sample(self, idx, indices, prob):
        if not self.pair_sampling:
            self.X[idx, indices] = 1
            self.actual_sample_prob[idx] = prob
        else:
            self.X[2*idx, indices] = 1
            self.actual_sample_prob[2*idx] = prob
            complement_indices = np.setdiff1d(np.arange(self.n), indices)
            self.X[2*idx + 1, complement_indices] = 1
            self.actual_sample_prob[2*idx + 1] = prob
       
    def sample_with_replacement(self, num_samples):
        self.X = np.zeros((num_samples, self.n), dtype=float)
        self.actual_sample_prob = np.zeros(num_samples, dtype=float)
        elements = np.arange(self.n)
        idx = 0
        sample_dist_size = [prob * scipy.special.binom(self.n, size) for size, prob in zip(self.valid_sizes, self.sample_dist)]
        sample_dist_size = np.array(sample_dist_size) / np.sum(sample_dist_size)
        num_pairs = num_samples // 2 if self.pair_sampling else num_samples
        sampled_sizes = self.gen.choice(self.valid_sizes, num_pairs, p=sample_dist_size)
        
        for idx, size in enumerate(sampled_sizes):
            indices = self.gen.choice(elements, size=size, replace=False)
            self.add_sample(idx, indices, self.sample_dist[size - self.adjust])
        
    def find_bernoulli_constant(self, num_samples):
        C = 1/2**self.n
        sample_dist_C = np.minimum(self.sample_dist * C, np.ones_like(self.sample_dist))
        m = min(num_samples, 2**self.n - 2 * self.adjust) 
        L = 1 / 100**self.n
        R = 100**self.n
        binoms = scipy.special.comb(self.n, self.valid_sizes)
        expected = np.sum(sample_dist_C * binoms)
        while round(expected) != m and R-L > 1e-6:
            if expected < m: L = C
            else: R = C
            C = (L + R) / 2
            sample_dist_C = np.minimum(self.sample_dist * C, np.ones_like(self.sample_dist))
            expected = np.sum(sample_dist_C * binoms)
        return C
    
    def sample_without_replacement(self, num_samples):
        C = self.find_bernoulli_constant(num_samples)
        sample_dist_C = np.minimum(self.sample_dist * C, np.ones_like(self.sample_dist))
        num_samples = np.sum([
            round(prob * scipy.special.binom(self.n, s)) for s, prob in zip(self.valid_sizes, sample_dist_C)
        ])
        num_samples = (num_samples // 2) * 2 if self.pair_sampling else num_samples
        self.X = np.zeros((num_samples, self.n), dtype=float)
        self.actual_sample_prob = np.zeros(num_samples, dtype=float)
        
        idx = 0
        for s, prob in zip(self.valid_sizes, sample_dist_C):
            m_s = round(prob * scipy.special.binom(self.n, s))
            if self.pair_sampling and s == self.n // 2 and self.n % 2 == 0:
                combo_gen = combination_generator(self.gen, self.n-1, s-1, m_s // 2)
                for indices in combo_gen:
                    self.add_sample(idx, list(indices) + [self.n-1], prob)
                    idx += 1
                break
            combo_gen = combination_generator(self.gen, self.n, s, m_s)
            for indices in combo_gen:
                self.add_sample(idx, indices, prob)
                idx += 1
            if self.pair_sampling and s == self.n // 2: break

 
    def explain(self, explicand, num_samples, k=10):
        if num_samples >= 2**self.n:
            return enumeration_prob(self.model, self.baseline, explicand, self.weighting)        
        if self.pair_sampling:
            num_samples = num_samples // (2*k) * (2*k)
        else:
            num_samples = num_samples // k * k
        
        if self.reg_model_class == 'linear' and self.weighting in ['shapley', 'beta_shapley_1_1']:
            self.v1 = self.model.predict(explicand)
            self.v0 = self.model.predict(self.baseline)
            
        groups = []
        if self.weighting in [] and self.reg_model_class == 'linear':
            return self.explain_special_linear(explicand, num_samples)
        
        else:
            self.sample_without_replacement(num_samples)
            num_samples = self.X.shape[0]
            sizes = self.X.sum(axis=1).astype(int)
            model_input = self.baseline * (1 - self.X) + explicand * self.X
            y = self.model.predict(model_input)
            # Randomly partition num_samples indices into k groups while keeping pairs together
            if self.pair_sampling:
                pair_ids = self.gen.permutation(num_samples // 2)
                shuffled_indices = np.reshape(np.column_stack((2 * pair_ids, 2 * pair_ids + 1)), -1)
            else:
                shuffled_indices = self.gen.permutation(shuffled_indices)
            for i in range(k):
                indices = shuffled_indices[i * (num_samples // k):(i + 1) * (num_samples // k)]
                group = {
                    'sampled' : self.X[indices],
                    'y' : y[indices],
                    'prob_sampled' : self.actual_sample_prob[indices],
                    'sizes' : self.X[indices].sum(axis=1).astype(int)
                }
                feature_names = [f'feature_{j}' for j in range(self.n)]
                X_train = pd.DataFrame(self.X[~np.isin(np.arange(num_samples), indices)], columns=feature_names)
                X_test = pd.DataFrame(self.X[indices], columns=feature_names)
                y_train = y[~np.isin(np.arange(num_samples), indices)]
                
                # For leverage SHAP
                cache = {
                    'prob_sampled': self.actual_sample_prob[~np.isin(np.arange(num_samples), indices)],
                    'v1': self.v1,
                    'v0': self.v0,
                    'p': self.p
                }
                reg_model, reg_phi = get_fit(
                    X_train, y_train, self.weighting, self.reg_model_class, cache
                )
                reg_pred = reg_model.predict(X_test)
                group['phi_est'] = reg_phi
                group['reg_pred'] = reg_pred
                groups.append(group)
 
        phi = np.zeros(self.n)
        for group in groups:

            sizes, reg_pred = group['sizes'], group['reg_pred']

            for idx in range(self.n):
                idx_contained = (group['sampled'][:, idx] == 1)
                not_contained = ~idx_contained
                mean_with = mean_without = 0
                if np.any(idx_contained):
                    mean_with = (
                        (group['y'][idx_contained] - reg_pred[idx_contained]) * self.p[sizes[idx_contained]-1]
                        / group['prob_sampled'][idx_contained]
                    ).mean()
                if np.any(not_contained):
                    mean_without = (
                        (group['y'][not_contained] - reg_pred[not_contained]) * self.p[sizes[not_contained]] 
                        / group['prob_sampled'][not_contained]
                    ).mean()
                phi[idx] += (group['phi_est'][idx] + mean_with - mean_without) / k
                if self.reg_model_class == 'linear' and self.weighting in ['shapley', 'beta_shapley_1_1']:
                    phi[idx] += (self.v1 - group['phi_est'].sum()) / self.n - (self.v0) / self.n

        return phi
        
    def explain_special_linear(self, explicand, num_samples):

        if self.weighting in ['shapley', 'beta_shapley_1_1']:
            constrain_reg = True
        elif self.weighting in ['banzhaf', 'weighted_banzhaf_0.5']:
            constrain_reg = False
        else:
            raise ValueError("weighting must be 'shapley', 'beta_shapley_1_1', 'banzhaf' or 'weighted_banzhaf_0.5'")
        
        LinearEstimator = UniversalRegression(
            self.model, self.baseline, self.weighting, with_replace=False, constrain_reg=constrain_reg,
        )

        phi_est = LinearEstimator.explain(explicand, num_samples)
        sampled = LinearEstimator.sampled
        y = LinearEstimator.y
        sizes = LinearEstimator.sizes
        prob_sampled = LinearEstimator.prob_sampled
        phi = np.zeros(self.n)

        pred = (sampled @ phi_est)

        for i in range(self.n):
            i_contained = (sampled[:, i] == 1)
            i_contained_weighting = self.p[sizes[i_contained]-1]
            not_contained_weighting = self.p[sizes[~i_contained]]

            phi[i] = (                
                phi_est[i] +
                (
                    (y[i_contained] - pred[i_contained]) * i_contained_weighting / prob_sampled[i_contained]
                 ).mean() -
                (
                    (y[~i_contained] - pred[~i_contained]) * not_contained_weighting / prob_sampled[~i_contained]
                ).mean()
            )
            if constrain_reg:
                phi[i] += (LinearEstimator.v1 - phi_est.sum())/self.n - (LinearEstimator.v0)/self.n

        return phi

class LinearMSR(UniversalMSR):
    def __init__(self, model, baseline, weighting):
        super().__init__(model, baseline, weighting, reg_model_class='linear')

class TreeMSR(UniversalMSR):
    def __init__(self, model, baseline, weighting):
        super().__init__(model, baseline, weighting, reg_model_class='tree')

class Tree(UniversalMSR):
    def __init__(self, model, baseline, weighting):
        super().__init__(model, baseline, weighting, reg_model_class='tree', return_direct=True)

class MSR(UniversalMSR):
    def __init__(self, model, baseline, weighting):
        super().__init__(model, baseline, weighting, reg_model_class=False)

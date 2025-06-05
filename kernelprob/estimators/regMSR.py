import numpy as np
import scipy
from scipy.special import comb
import scipy.special
import pandas as pd

import sklearn.ensemble
import sklearn.linear_model
from ..exact.treeshap import tree_shap
from ..exact.treeprob import tree_prob
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

def get_fit(X_flat, y_flat, weighting, regression_adj):
    n = X_flat.shape[1]
    if regression_adj is False:
        reg_model = NullModel()
        phi_method = lambda reg_model : np.zeros(n)
    elif regression_adj == "linear":
        reg_model = sklearn.linear_model.LinearRegression()
        phi_method = lambda reg_model: reg_model.coef_
    elif regression_adj == "tree":
        #reg_model = sklearn.ensemble.RandomForestRegressor()
        reg_model = xgboost.XGBRegressor()
#        phi_method = lambda reg_model: tree_prob(np.zeros((1,n)), np.ones((1,n)), reg_model, weighting)[0].squeeze()
        if weighting in ['shapley', 'beta_shapley_1_1']:
            phi_method = lambda reg_model: tree_shap(np.zeros((1,n)), np.ones((1,n)), reg_model)[0].squeeze()
        else:
            phi_method = lambda reg_model: tree_prob(np.zeros((1,n)), np.ones((1,n)), reg_model, weighting)[0].squeeze()
    else:
        raise ValueError("regression_adjustment must be False, 'linear', or 'tree'")

    reg_model.fit(X_flat.to_numpy(), y_flat)
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
        self.sample_prob = self.p * scipy.special.binom(self.n-1, np.arange(self.n))
        
        sample_prob = []
        for size in range(self.n+1):
            prob = 0
            if size > 0:
                prob += self.p[size-1]**2 * size
            if size < self.n:
                prob += self.p[size]**2 * (self.n - size)
            sample_prob.append(
                np.sqrt(prob) * scipy.special.binom(self.n, size)
            )
        self.sample_prob = np.array(sample_prob) / np.sum(sample_prob)

        self.return_direct = return_direct
       
    def sample_with_replacement(self, num_samples):
        self.X = np.zeros((num_samples, self.n), dtype=float)
        self.actual_sample_prob = np.zeros(num_samples, dtype=float)

        elements = np.arange(self.n)
        idx = 0
        sizes = np.arange(self.n+1)
        sampled_sizes = self.gen.choice(sizes, num_samples, p=self.sample_prob)
        
        for idx, size in enumerate(sampled_sizes):
            indices = self.gen.choice(elements, size=size, replace=False)

            self.X[idx, indices] = 1
            self.actual_sample_prob[idx] = self.sample_prob[size] / scipy.special.binom(self.n, size)
    
    def sample_without_replacement(self, num_samples):
        pass
 
    def explain(self, explicand, num_samples, k=10):
        num_samples = num_samples // k * k  # Ensure num_samples is divisible by k
        phi_hats = {}
        if self.weighting in ['shapley', 'beta_shapley_1_1', 'banzhaf', 'weighted_banzhaf_0.5'] and self.reg_model_class == 'linear':
            return self.explain_special_linear(explicand, num_samples)
        
        else:
            self.sample_with_replacement(num_samples)
            model_input = self.baseline * (1 - self.X) + explicand * self.X
            y = self.model.predict(model_input)
            for i in range(k):
                indices = np.arange(i * (num_samples // k), (i + 1) * (num_samples // k))
                phi_hats[i] = {
                    'sampled' : self.X[indices],
                    'y' : y[indices],
                    'prob_sampled' : self.actual_sample_prob[indices],
                    'sizes' : self.X[indices].sum(axis=1).astype(int)
                }
            feature_names = [f"f{i}" for i in range(model_input.shape[1])]
            for i in range(k):
                # Fit regression model on all other samples
                # and predict on the i-th sample
                phi_hats[i]['phi_est'] = np.zeros(self.n)
                phi_hats[i]['reg_pred'] = np.zeros(self.X.shape[0])
                not_i = [j for j in range(k) if j != i]
                X_train = pd.DataFrame(np.vstack([phi_hats[j]['sampled'] for j in not_i]), columns=feature_names)
                y_train = np.hstack([phi_hats[j]['y'] for j in not_i])
                X_test = pd.DataFrame(phi_hats[i]['sampled'], columns=feature_names)
                reg_model, reg_phi = get_fit(X_train, y_train, self.weighting, self.reg_model_class)
                reg_pred = reg_model.predict(X_test)
                phi_hats[i]['phi_est'] = reg_phi
                phi_hats[i]['reg_pred'] = reg_pred
 
        phis = {i : np.zeros(self.n) for i in range(k)}
        for i in range(k):
            sizes = phi_hats[i]['sizes']
            sampled = phi_hats[i]['sampled']
            y = phi_hats[i]['y']
            prob_sampled = phi_hats[i]['prob_sampled']
            phi_est = phi_hats[i]['phi_est']
            reg_pred = phi_hats[i]['reg_pred']

            for idx in range(self.n):
                idx_contained = (sampled[:, idx] == 1)             
                not_contained = ~idx_contained
                mean_with = mean_without = 0
                if np.any(idx_contained):
                    mean_with = (
                        (y[idx_contained] - reg_pred[idx_contained]) * self.p[sizes[idx_contained]-1] / prob_sampled[idx_contained]
                    ).mean()
                if np.any(not_contained):
                    mean_without = (
                        (y[not_contained] - reg_pred[not_contained]) * self.p[sizes[not_contained]] / prob_sampled[not_contained]
                    ).mean()
                phis[i][idx] = phi_est[idx] + mean_with - mean_without

        phi = np.mean([phis[i] for i in range(k)], axis=0)
        return phi

    def explain2(self, explicand, num_samples):
        self.num_samples = num_samples // 2 * 2
        self.pair_sampling = False
        if self.weighting in ['shapley', 'beta_shapley_1_1', 'banzhaf', 'weighted_banzhaf_0.5'] and self.reg_model_class == 'linear':
            return self.explain_special_linear(explicand, num_samples) 

        self.sample_with_replacement(self.num_samples)
        
        model_input = self.baseline * (1 - self.X) + explicand * self.X
        y = self.model.predict(model_input)

        # randomly sample m/2 training indices
        train_indices = self.gen.choice(self.num_samples, self.num_samples//2, replace=False)
        if self.split_training:
            X_train, y_train = self.X[train_indices], y[train_indices]
            X_test, y_test = self.X[~train_indices], y[~train_indices]
        else:
            X_train, y_train = X_test, y_test = self.X, y

        feature_names = [f"f{i}" for i in range(model_input.shape[1])]
        X_train_df = pd.DataFrame(X_train, columns=feature_names)
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        reg_model, reg_phi = get_fit(X_train_df, y_train, self.weighting, self.reg_model_class)        
        reg_pred = reg_model.predict(X_test_df)

        if self.return_direct:
            return reg_phi
 
        phi = np.zeros(self.n)
        for i in range(self.n):
            i_contained = (X_test[:, i] == 1)             
            not_contained = ~i_contained
            sizes = X_test.sum(axis=1).astype(int)
            if self.split_training:
                actual_sample_prob_test = self.actual_sample_prob[~train_indices]
            else:
                actual_sample_prob_test = self.actual_sample_prob
            #if self.split_samples:
            #    offset = np.zeros(len(self.X), dtype=int)
            #    offset[self.num_samples//2:] = 1
            #    i_contained = (self.X[:,i]==1) & (offset == 1)
            #    not_contained = (self.X[:,i]==0) & (offset== 0)
            phi[i] = (
                reg_phi[i] + 
                (
                    (y_test[i_contained] - reg_pred[i_contained]) * self.p[sizes[i_contained]-1] / actual_sample_prob_test[i_contained]
                 ).mean() -
                (
                    (y_test[not_contained] - reg_pred[not_contained]) * self.p[sizes[not_contained]] / actual_sample_prob_test[not_contained]
                ).mean()
            )

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

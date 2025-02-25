import numpy as np
import scipy
import sklearn.ensemble

from ..base_estimator import BaseEstimator
from ..utils.p_generator import get_p
from ..exact.treeprob import tree_prob

class RegressionAdjustmentEstimator(BaseEstimator):
    def __init__(self, model, baseline: np.ndarray, weighting: str):
        super().__init__(model, baseline, weighting)

        self.n = self.baseline.shape[1] if self.baseline.ndim > 1 else self.baseline.size

        self.p = get_p(self.n, weighting)
        # pair sample if p is symmetric
        self.pair_sampling = np.allclose(self.p, self.p[::-1])
        #self.sample_prob = np.ones(self.n)
        self.sample_prob = self.p * scipy.special.binom(self.n-1, np.arange(self.n))
        self.sample_prob /= np.sum(self.sample_prob)
        self.rng = np.random.default_rng()

    def explain(self, explicand: np.ndarray, num_samples: int) -> np.ndarray:
        num_samples = num_samples // 2
        self.nue_avg = num_samples // self.n
        phi = np.zeros_like(self.baseline, dtype=float)

        X = np.zeros((self.nue_avg * 2 * self.n, self.n), dtype=float)
        y = np.zeros(self.nue_avg * 2 * self.n, dtype=float)
        sign = np.ones(self.nue_avg * 2 * self.n, dtype=float)
        weights = np.ones(self.nue_avg * 2 * self.n, dtype=float)

        for i in range(self.n):
            except_i = np.delete(np.arange(self.n), i)
            offset = i * self.nue_avg * 2

            for S_idx in range(self.nue_avg):
                size = self.rng.choice(self.n, p=self.sample_prob)
                indices = self.rng.choice(except_i, size=size, replace=False)
                indices_with_i = np.append(indices, i)

                X[offset + 2*S_idx, indices] = 1
                X[offset + 2*S_idx + 1, indices_with_i] = 1

                sign[offset + 2 * S_idx] = -1  # Assign -1 for the without_i subset

                weight = self.p[size] * scipy.special.binom(self.n-1, size) / self.sample_prob[size]
                weights[offset + 2 * S_idx] = weight
                weights[offset + 2 * S_idx + 1] = weight
        
        model_input = self.baseline * (1 - X) + explicand * X
        y = self.model.predict(model_input)

        # Fit on all data
        reg_model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, max_features=self.n)
        reg_model.fit(model_input, y)
        reg_model_phi = tree_prob(self.baseline, explicand, reg_model, self.weighting)[0].squeeze()

        for i in range(self.n):
            offset = i * self.nue_avg * 2
            end = offset + 2 * self.nue_avg
            # Exclude everything from offset to end from X            
            #mask = np.ones(len(X), dtype=bool)
            #mask[offset:end] = False
            #reg_model.fit(X[mask], y[mask])
            #reg_model_phi = tree_prob(self.baseline, explicand, reg_model, self.weighting)[0].squeeze()

            reg_pred = reg_model.predict(model_input[offset:end])
            reg_correction = reg_model_phi[i] 

            phi[:,i] = np.sum(y[offset:end] * sign[offset:end] * weights[offset:end], axis=0) / self.nue_avg
            phi[:,i] -= np.sum(reg_pred * sign[offset:end] * weights[offset:end], axis=0) / self.nue_avg
            phi[:,i] += np.sum(reg_correction / 2 * weights[offset:end], axis=0) / self.nue_avg

#            model_input = self.baseline * (1 - model_input) + explicand * model_input
#            model_output = self.model.predict(model_input)

#            phi[:, i] = np.sum(model_output * sign * weights, axis=0) / self.nue_avg
        
        return phi
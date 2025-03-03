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
        self.per_S = 2 if not self.pair_sampling else 4

        #self.sample_prob = np.ones(self.n)
        self.sample_prob = self.p * scipy.special.binom(self.n-1, np.arange(self.n))
        self.sample_prob /= np.sum(self.sample_prob)
        self.rng = np.random.default_rng()
    
    def add_one_sample(self, idx, indices, weight, sign):
        self.X[idx, indices] = 1
        self.sign[idx] = sign
        self.weights[idx] = weight
    
    def sample_with_replacement(self, num_samples):
        for i in range(self.n):
            except_i = np.delete(np.arange(self.n), i)
            offset = i * self.nue_avg * self.per_S

            for S_idx in range(self.nue_avg):
                size = self.rng.choice(self.n, p=self.sample_prob)
                indices = self.rng.choice(except_i, size=size, replace=False)
                indices_with_i = np.append(indices, i)
                weight = self.p[size] * scipy.special.binom(self.n-1, size) / self.sample_prob[size]

                self.add_one_sample(offset + self.per_S * S_idx, indices, weight, -1)
                self.add_one_sample(offset + self.per_S * S_idx + 1, indices_with_i, weight, 1)
            
                if self.pair_sampling:
                    indices_complement = np.delete(np.arange(self.n), indices_with_i)
                    #size2 = len(indices_complement)
                    #weight2 = self.p[size2] * scipy.special.binom(self.n-1, size2) / self.sample_prob[size2]
                    #assert np.isclose(weight, weight2)

                    indices_complement_with_i = np.append(indices_complement, i)

                    self.add_one_sample(offset + self.per_S * S_idx + 2, indices_complement, weight, -1)
                    self.add_one_sample(offset + self.per_S * S_idx + 3, indices_complement_with_i, weight, 1)

    def explain(self, explicand: np.ndarray, num_samples: int) -> np.ndarray:
        
        self.nue_avg = (num_samples // self.per_S) // self.n
        num_samples = self.nue_avg * self.per_S * self.n

        self.X = np.zeros((num_samples, self.n), dtype=float)
        self.y = np.zeros(num_samples, dtype=float)
        self.sign = np.ones(num_samples, dtype=float)
        self.weights = np.ones(num_samples, dtype=float)

        self.sample_with_replacement(num_samples)
        
        model_input = self.baseline * (1 - self.X) + explicand * self.X
        y = self.model.predict(model_input)

        # Fit on all data
        reg_model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, max_features=self.n)
        reg_model.fit(model_input, y)
        reg_model_phi = tree_prob(self.baseline, explicand, reg_model, self.weighting)[0].squeeze()

        phi = np.zeros(self.n)

        for i in range(self.n):
            offset = i * self.nue_avg * self.per_S
            end = offset + self.per_S * self.nue_avg
            # Exclude everything from offset to end from X            
            #mask = np.ones(len(X), dtype=bool)
            #mask[offset:end] = False
            #reg_model.fit(X[mask], y[mask])
            #reg_model_phi = tree_prob(self.baseline, explicand, reg_model, self.weighting)[0].squeeze()

            reg_pred = reg_model.predict(model_input[offset:end])
            reg_correction = reg_model_phi[i] 

            phi[i] = np.mean(y[offset:end] * self.sign[offset:end] * self.weights[offset:end], axis=0) * 2
            phi[i] -= np.mean(reg_pred * self.sign[offset:end] * self.weights[offset:end], axis=0) * 2
            phi[i] += np.mean(reg_correction / 2 * self.weights[offset:end], axis=0) * 2
        
        return phi
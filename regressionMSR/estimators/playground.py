import numpy as np
import scipy
import sklearn.ensemble

from ..base_estimator import BaseEstimator
from ..utils.p_generator import get_p
from ..exact.treeprob import tree_prob
from ..exact.enumeration import combination_generator
from ..estimators.reg import UniversalRegression

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

class MSRAdjusted(BaseEstimator):
    def __init__(self, model, baseline: np.ndarray, weighting: str, regression_adjust='tree', use_median=False):
        super().__init__(model, baseline, weighting)

        self.n = self.baseline.shape[1] if self.baseline.ndim > 1 else self.baseline.size
        self.p = get_p(self.n, weighting)
        self.regression_adjust = regression_adjust
        self.use_median = use_median
        
        # pair sample if p is symmetric
        self.pair_sampling = np.allclose(self.p, self.p[::-1])
        self.pair_sampling = False

        self.sample_prob = self.p * scipy.special.binom(self.n-1, np.arange(self.n))
        self.rng = np.random.default_rng()
    
    def add_sample(self, idx, indices, bigger_subset):
        self.X[idx, indices] = 1
        self.bigger_subset[idx] = bigger_subset
        self.sizes[idx] = len(indices)
    
    def sample_with_replacement(self):

        elements = np.arange(self.n)

        idx = 0
        if self.pair_sampling:
            self.num_samples = self.num_samples // 2
        for offset in [0,1]:
            sizes = np.arange(self.n) + offset
            sampled_sizes = self.rng.choice(sizes, self.num_samples//2, p=self.sample_prob)
            
            for S_idx, size in enumerate(sampled_sizes):
                indices = self.rng.choice(elements, size=size, replace=False)

                self.add_sample(idx, indices, offset)
                idx += 1
                
                if self.pair_sampling:
                    indices_complement = np.delete(np.arange(self.n), indices)
                    self.add_sample(idx, indices_complement, offset)
                    idx += 1


    def explain(self, explicand: np.ndarray, num_samples: int, return_direct=False) -> np.ndarray:
        self.num_samples = (num_samples // 2) * 2
        self.X = np.zeros((self.num_samples, self.n), dtype=float)
        self.bigger_subset = np.ones((self.num_samples), dtype=float)
        self.sizes = np.zeros((self.num_samples), dtype=int)

        self.sample_with_replacement()
        
        model_input = self.baseline * (1 - self.X) + explicand * self.X
        y = self.model.predict(model_input)

        reg_model, reg_phi = get_fit(self.X, y, self.weighting, self.regression_adjust)
        reg_pred = reg_model.predict(self.X)

        if return_direct: return reg_model
 
        phi = np.zeros(self.n)

        for i in range(self.n):
            i_contained = (self.X[:, i] == 1)
            
            phi[i] = (
                reg_phi[i] + 
                (
                    (y[i_contained] - reg_pred[i_contained])
                 ).mean() -
                (
                    (y[~i_contained] - reg_pred[~i_contained])
                ).mean()
            )

        return phi

class TreeMSRAdjusted(MSRAdjusted):
    def __init__(self, model, baseline: np.ndarray, weighting: str):
        super().__init__(model, baseline, weighting, regression_adjust="tree", use_median=False)

class LinearMSRAdjusted(MSRAdjusted):
    def __init__(self, model, baseline: np.ndarray, weighting: str):
        super().__init__(model, baseline, weighting, regression_adjust="linear", use_median=False)

class Sampler():
    def __init__(self, n, base_probs, valid_sizes, pair_sampling):
        self.n = n
        self.base_probs = base_probs
        self.valid_sizes = valid_sizes
        self.pair_sampling = pair_sampling
        self.gen = np.random.Generator(np.random.PCG64())

    def add_one_sample(self, idx, indices):
        if not self.pair_sampling:
            self.sampled[idx, indices] = 1
        else:            
            self.sampled[2*idx, indices] = 1
            indices_complement = np.array([i for i in range(self.n) if i not in indices])
            self.sampled[2*idx+1, indices_complement] = 1
    
    def get_probs(self, s):
        return self.probs[s - self.valid_sizes[0]]
    
    def update_prob(self):
        lev = self.base_probs[self.valid_sizes-self.valid_sizes[0]]
        if self.pair_sampling:
            lev += self.base_probs[self.n - self.valid_sizes]
        self.probs = np.minimum(self.C * lev, np.ones_like(lev))
    
    def find_constant_for_bernoulli(self):
        # Choose C so that sampling without replacement from min(1, C*prob) gives the same expected number of samples
        self.C = 1/2**self.n
        self.update_prob()
        max_samples = 2**self.n
        m = min(self.num_samples, max_samples-1)
        # Find C with binary search
        L = 1 / 100**self.n
        R = 100 ** self.n
        binoms = scipy.special.binom(self.n, self.valid_sizes)
        expected = (binoms@self.probs).sum()
        while round(expected) != m:
            if expected < m: L = self.C
            else: R = self.C
            self.C = (L + R) / 2
            self.update_prob() 
            expected = (binoms@self.probs).sum()

    def sample_without_replacement(self, num_samples):
        self.num_samples = (num_samples // 2) * 2
        self.find_constant_for_bernoulli()
        m_s_lookup = {}
        for s, prob in zip(self.valid_sizes, self.probs):
            # DETERMINISTIC SAMPLING 
            m_s = int(np.round(prob * scipy.special.binom(self.n, s)))
            m_s_lookup[s] = m_s
            if self.pair_sampling and s == self.n // 2: # Already sampled all larger sets with the complement 
                    if self.n % 2 == 0: # Special handling for middle set size if n is even
                        m_s_lookup[s] = m_s // 2
                    break

        sampled_m = np.sum([m_s for m_s in m_s_lookup.values()])
        num_rows = sampled_m if not self.pair_sampling else sampled_m * 2
        self.sampled = np.zeros((num_rows, self.n))
        idx = 0
        for s, m_s in m_s_lookup.items(): 
            if self.pair_sampling and s == self.n // 2 and self.n % 2 == 0:
                # Partition the all middle sets into two
                # based on whether the combination contains n-1
                combo_gen = combination_generator(self.gen, self.n - 1, s-1, m_s)
                for indices in combo_gen:
                    self.add_one_sample(idx, list(indices) + [self.n-1])
                    idx += 1
            else:
                combo_gen = combination_generator(self.gen, self.n, s, m_s)
                for indices in combo_gen:
                    self.add_one_sample(idx, list(indices))
                    idx += 1
    
        sizes = np.sum(self.sampled, axis=1).astype(int)
        prob_sampled = self.get_probs(sizes)
        return self.sampled, prob_sampled

class UniversalMSR(BaseEstimator):
    def __init__(self, model, baseline, weighting, reg_model_class=False):
        super().__init__(model, baseline, weighting)
        self.n = self.baseline.shape[1]
        self.p = get_p(self.n, weighting)
        self.model = model
        self.baseline = baseline
        self.gen = np.random.Generator(np.random.PCG64())
        self.reg_model_class = reg_model_class

        self.pair_sampling = np.allclose(self.p, self.p[::-1])
 
    def explain(self, explicand, num_samples):
        self.num_samples = num_samples // 2 * 2
        self.pair_sampling = False
        if self.weighting in ['shapley', 'betashapley_1_1'] and self.reg_model_class == 'linear':
            return self.explain_linear_shapley(explicand, num_samples)
        

        # Sampler
        if False: # Sample together
            valid_sizes = np.arange(0, self.n+1) 
            self.base_probs = np.zeros(self.n+1)
            for s in valid_sizes:
                if s == 0:
                    self.base_probs[s] = self.p[s] / self.n
                elif s == self.n:
                    self.base_probs[s] = self.p[s-1] / self.n
                else:
                    self.base_probs[s] = (self.p[s-1] + self.p[s]) / ((self.n-s) * s)
            
            self.Sampler = Sampler(
                self.n, self.base_probs, valid_sizes, self.pair_sampling
            ) 

            sampled, prob_sampled = self.Sampler.sample_without_replacement(self.num_samples)
        
        else:
            self.Sampler = Sampler(
                self.n, self.p, np.arange(self.n), False
            )
            sampled1, prob_sampled1 = self.Sampler.sample_without_replacement(self.num_samples//2)
            self.Sampler = Sampler(
                self.n, self.p, np.arange(1, self.n+1), False
            )
            sampled2, prob_sampled2 = self.Sampler.sample_without_replacement(self.num_samples//2)
            sampled = np.concatenate((sampled1, sampled2), axis=0)
            prob_sampled = np.concatenate((prob_sampled1, prob_sampled2), axis=0)
        
        #LeverageSHAP = UniversalRegression(
        #    self.model, self.baseline, self.weighting, with_replace=False, constrain_reg=True,
        #)
        #reg_phi = LeverageSHAP.explain(explicand, num_samples)
        #sampled = LeverageSHAP.sampled
        #y = LeverageSHAP.y
        #sizes = LeverageSHAP.sizes
        #prob_sampled = LeverageSHAP.prob_sampled

        #reg_pred = (sampled @ reg_phi)


        model_input = self.baseline * (1 - sampled) + explicand * sampled
        y = self.model.predict(model_input)

        #sizes = np.sum(self.sampled, axis=1).astype(int)

 
        
        reg_model, reg_phi = get_fit(sampled, y, self.weighting, self.reg_model_class)
        reg_pred = reg_model.predict(sampled)

        #reg_model, reg_phi = get_fit(model_input, y, self.weighting, self.reg_model_class)
        #reg_pred = reg_model.predict(model_input)

        print('sampled:', sampled.shape)
        print(sampled)

        phi = np.zeros(self.n)
        sizes = np.sum(sampled, axis=1).astype(int)

        for i in range(self.n):
            i_contained = (sampled[:, i] == 1)
            i_contained_weighting = self.p[sizes[i_contained]-1]
            not_contained_weighting = self.p[sizes[~i_contained]]

            phi[i] = (
                reg_phi[i] +
                (
                    (y[i_contained] - reg_pred[i_contained]) * i_contained_weighting / prob_sampled[i_contained]
                 ).mean() -
                (
                    (y[~i_contained] - reg_pred[~i_contained]) * not_contained_weighting / prob_sampled[~i_contained]
                ).mean()
            )
        
        return phi


    def explain_linear_shapley(self, explicand, num_samples):
        LeverageSHAP = UniversalRegression(
            self.model, self.baseline, self.weighting, with_replace=False, constrain_reg=True,
        )
        phi_est = LeverageSHAP.explain(explicand, num_samples)
        sampled = LeverageSHAP.sampled
        y = LeverageSHAP.y
        sizes = LeverageSHAP.sizes
        prob_sampled = LeverageSHAP.prob_sampled
        phi = np.zeros(self.n)

        pred = (sampled @ phi_est)

        for i in range(self.n):
            i_contained = (sampled[:, i] == 1)
            i_contained_weighting = self.p[sizes[i_contained]-1]
            not_contained_weighting = self.p[sizes[~i_contained]]

            phi[i] = (
                (LeverageSHAP.v1 - phi_est.sum())/self.n -(LeverageSHAP.v0)/self.n +
                phi_est[i] +
                (
                    (y[i_contained] - pred[i_contained]) * i_contained_weighting / prob_sampled[i_contained]
                 ).mean() -
                (
                    (y[~i_contained] - pred[~i_contained]) * not_contained_weighting / prob_sampled[~i_contained]
                ).mean()
            )

        return phi

class MSRLinear(UniversalMSR):
    def __init__(self, model, baseline, weighting):
        super().__init__(model, baseline, weighting, reg_model_class='linear')

class MSRTree(UniversalMSR):
    def __init__(self, model, baseline, weighting):
        super().__init__(model, baseline, weighting, reg_model_class=False)

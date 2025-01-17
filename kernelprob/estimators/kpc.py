import numpy as np
import scipy
import scipy.special
import math

from ..base_estimator import BaseEstimator
from ..utils.p_generator import get_p
from .est_utils import combination_generator

class KernelProbCEstimator(BaseEstimator):
    def __init__(
        self,
        model,
        baseline,
        weighting,
        paired_sampling=True, # only paired sampling
        leverage_sampling=True, # only leverage score
        bernoulli_sampling=True, # only bernoulli sampling
    ):
        """
        Additional Parameters
        ----------
        paired_sampling : bool
            Whether to sample sets in pairs (S and its complement).
        leverage_sampling : bool
            If True, activates a special weighting scheme inside the sampling functions.
        bernoulli_sampling : bool
            If True, sample sets without replacement via a Bernoulli-like process.
        """
        super().__init__(model, baseline, weighting)
        self.p = get_p(baseline.shape[1], weighting)
        self.model = model
        self.baseline = baseline
        self.paired_sampling = paired_sampling
        self.n = self.baseline.shape[1] # Number of features
        self.gen = np.random.Generator(np.random.PCG64())

        self.leverage_score = lambda s : (self.p[s] + self.p[s-1]) * s * (self.n - s) 
    
    def get_row_prob(self, s):
        # Convert s to int
        total_leverage_score = self.leverage_score(s) + self.leverage_score(self.n-s)
        return min(1, self.C * total_leverage_score)
    
    def expected_samples(self):
        return np.sum([scipy.special.binom(self.n, s) * self.get_row_prob(s) for s in range(1, self.n)])
    
    def add_one_sample(self, idx, indices, weight):
        #indices = sorted(indices)
        #if tuple(indices) in self.used_indices: return
        #self.used_indices.add(tuple(indices))
        if not self.paired_sampling:
            self.SZ_binary[idx, indices] = 1
            self.kernel_weights.append(weight)
        else:
            indices_complement = np.array([i for i in range(self.n) if i not in indices])
            self.SZ_binary[2*idx, indices] = 1
            self.kernel_weights.append(weight)
            self.SZ_binary[2*idx+1, indices_complement] = 1
            self.kernel_weights.append(weight)

    
    def sample_with_replacement(self):
        # Not implemented
        pass 
        #self.SZ_binary = np.zeros((self.num_samples, self.n))
        #valid_sizes = np.array(list(range(1, self.n)))
        #prob_sizes = self.sample_weight(valid_sizes)
        #prob_sizes = prob_sizes / np.sum(prob_sizes)
        #num_sizes = self.num_samples if not self.paired_sampling else self.num_samples // 2
        #sampled_sizes = self.gen.choice(valid_sizes, num_sizes, p=prob_sizes)
        #for idx, s in enumerate(sampled_sizes):
        #    indices = self.gen.choice(self.n, s, replace=False)
        #    # weight = Pr(sampling this set) * w(s)
        #    weight = 1 / (self.sample_weight(s) * s * (self.n - s))
        #    self.add_one_sample(idx, indices, weight=weight)
    
    def find_constant_for_bernoulli(self):
        # Choose C so that sampling without replacement from min(1, C*prob) gives the same expected number of samples
        self.C = 1/2**self.n # Assume at least n - 1 samples
        m = min(self.num_samples, 2**self.n-2) # Maximum number of samples is 2^n -2
        # Efficiently find C with binary search
        L = 1 / 2**self.n
        R = scipy.special.binom(self.n, self.n // 2) * 4 ** self.n
        expected = self.expected_samples()
        while round(expected) != m:
            if expected < m: L = self.C
            else: R = self.C
            self.C = (L + R) / 2
            expected = self.expected_samples()
    
    def sample_without_replacement(self):
        self.find_constant_for_bernoulli()
        m_s_all = []
        expected = 0
        for s in range(1, self.n):
            expected += scipy.special.binom(self.n, s) * self.get_row_prob(s)
            # Sample from Binomial distribution with (n choose s) trials and probability min(1, 2*C*sample_weight(s) / (n choose s))
            prob = self.get_row_prob(s)
            try:
                m_s = self.gen.binomial(int(scipy.special.binom(self.n, s)), prob)
            except OverflowError: # If the number of samples is too large, assume the number of samples is the expected number
                m_s = int(prob * scipy.special.binom(self.n, s))
            if self.paired_sampling:
                if s == self.n // 2: # Already sampled all larger sets with the complement                    
                    if self.n % 2 == 0: # Special handling for middle set size if n is even
                        m_s_all.append(m_s // 2)
                    else: m_s_all.append(m_s)
                    break
            m_s_all.append(m_s)
        sampled_m = np.sum(m_s_all)
        num_rows = sampled_m if not self.paired_sampling else sampled_m * 2
        self.SZ_binary = np.zeros((num_rows, self.n))
        idx = 0
        for s, m_s in enumerate(m_s_all):
            s += 1
            prob = self.get_row_prob(s)
            # weight = 1/ prob * weighting in regression problem
            weight = 1 / prob * (self.p[s] + self.p[s-1])# * s * (self.n - s)
            if self.paired_sampling and s == self.n // 2 and self.n % 2 == 0:
                # Partition the all middle sets into two
                # based on whether the combination contains n-1
                combo_gen = combination_generator(self.gen, self.n - 1, s-1, m_s)
                for indices in combo_gen:
                    self.add_one_sample(idx, list(indices) + [self.n-1], weight = weight)
                    idx += 1
            else:
                combo_gen = combination_generator(self.gen, self.n, s, m_s)
                for indices in combo_gen:
                    self.add_one_sample(idx, list(indices), weight = weight)
                    idx += 1
    
    def explain(self, explicand, num_samples):
        self.explicand = explicand
        self.num_samples = int((num_samples -2 ) // 2) * 2
        self.kernel_weights = []
        
        # Sample
        self.sample_without_replacement()
        # A = Z P
        # b = v(z) - v0
        # (A^T S^T S A)^-1 A^T S^T S b + (v1 - v0) / n
        # (P^T Z^T S^T S Z P)^-1 P^T Z^T S^T S b + (v1 - v0) / n

        # Remove zero rows
        SZ_binary = self.SZ_binary[np.sum(self.SZ_binary, axis=1) != 0]
        v0, v1 = self.model.predict(self.baseline), self.model.predict(self.explicand)
        inputs = self.baseline * (1 - SZ_binary) + self.explicand * SZ_binary
        Sv = self.model.predict(inputs) - v0

       # Estimate constant term = 1/n sum_i phi_i
        # = 1/n sum_S (v(S)-v(emptyset)) (-p[|S|]*(n-|S|) + p[|S|-1] * |S|)
        # Compute probability each set size was sampled
        row_sum = np.sum(SZ_binary, axis=1).astype(int)
        prob_sampled = self.kernel_weights / (self.p[row_sum] + self.p[row_sum-1])
        sum_weighting = -self.p[row_sum] * (self.n - row_sum) + self.p[row_sum-1] * row_sum
        sum_phi = prob_sampled * sum_weighting @ Sv
        sum_phi = sum_phi + (v1-v0)*self.p[-1]*self.n
        
        Sb = Sv - row_sum * sum_phi / self.n

        # Projection matrix
        P = np.eye(self.n) - 1/self.n * np.ones((self.n, self.n))

        PZSSb = P @ SZ_binary.T @ np.diag(self.kernel_weights) @ Sb
        PZSSZP = P @ SZ_binary.T @ np.diag(self.kernel_weights) @ SZ_binary @ P
        PZSSZP_inv_PZSSb = np.linalg.lstsq(PZSSZP, PZSSb, rcond=None)[0]

        self.phi = PZSSZP_inv_PZSSb 
        self.phi += sum_phi / self.n

        return self.phi
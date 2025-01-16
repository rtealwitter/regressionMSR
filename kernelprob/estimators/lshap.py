import numpy as np
import scipy
import scipy.special
import math

from ..base_estimator import BaseEstimator
from .est_utils import combination_generator


class LeverageSHAPEstimator(BaseEstimator):
    def __init__(
        self,
        model,
        baseline,
        weighting,
        paired_sampling=False,
        leverage_sampling=False,
        bernoulli_sampling=False
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
        self.paired_sampling = paired_sampling
        self.leverage_sampling = leverage_sampling
        self.bernoulli_sampling = bernoulli_sampling

        self.n = self.baseline.shape[1]
        self.gen = np.random.Generator(np.random.PCG64())

        # Depending on leverage_sampling, set sample_weight:
        if not self.leverage_sampling:
            # sample_weight(s) = 1 / [s*(n-s)]
            self._sample_weight = lambda s: 1.0 / (s * (self.n - s)) if 0 < s < self.n else 0.0
        else:
            # sample_weight(s) = 1 (for s in [1..n-1])
            self._sample_weight = lambda s: 1.0

        self._reweight = lambda s: 1 / (
            self._sample_weight(s) * s * (self.n - s)
        ) if 0 < s < self.n else 0.0

        # Will hold the sampled set matrix (SZ_binary) and kernel weights
        self.SZ_binary = None
        self.kernel_weights = []

    def _add_one_sample(self, idx, indices, weight):
        if not self.paired_sampling:
            self.SZ_binary[idx, indices] = 1
            self.kernel_weights.append(weight)
        else:
            indices_complement = np.array([i for i in range(self.n) if i not in indices])
            self.SZ_binary[2*idx, indices] = 1
            self.kernel_weights.append(weight)
            self.SZ_binary[2*idx+1, indices_complement] = 1
            self.kernel_weights.append(weight)

    def _find_constant_for_bernoulli(self, max_C=1e10):
        """
        Binary search for a constant C so that expected #samples = self.num_samples.
        """
        # The maximum total samples we can handle is 2^n - 2
        m = min(self.num_samples, 2**self.n - 2)

        def expected_samples(C):
            return np.sum([min(
                    scipy.special.binom(self.n, s), 2 * C * self._sample_weight(s))
                for s in range(1, self.n)
                ])
        L = 1
        R = scipy.special.binom(self.n, self.n // 2) * self.n ** 2
        C = 1
        while True:
            curr = round(expected_samples(C))
            if curr == m:
                break
            if curr < m:
                L = C
            else:
                R = C
            C_new = (L + R) / 2
            if abs(C_new - C) < 1e-12:
                break
            C = C_new
        self.C = round(C)

    def _sample_with_replacement(self):
        self.SZ_binary = np.zeros((self.num_samples, self.n))
        valid_sizes = np.array(list(range(1, self.n)))
        prob_sizes = self._sample_weight(valid_sizes)
        prob_sizes = prob_sizes / np.sum(prob_sizes)
        num_sizes = self.num_samples if not self.paired_sampling else self.num_samples // 2
        sampled_sizes = self.gen.choice(valid_sizes, num_sizes, p=prob_sizes)
        for idx, s in enumerate(sampled_sizes):
            indices = self.gen.choice(self.n, s, replace=False)
            # weight = Pr(sampling this set) * w(s)
            weight = 1 / (self._sample_weight(s) * s * (self.n - s))
            self._add_one_sample(idx, indices, weight=weight)

    def _sample_without_replacement(self):
        self._find_constant_for_bernoulli()
        m_s_all = []
        for s in range(1, self.n):
            # Sample from Binomial distribution with (n choose s) trials and probability min(1, C*sample_weight(s) / (n choose s))
            prob = min(1, 2*self.C * self._sample_weight(s) / scipy.special.binom(self.n, s))
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
            prob = min(1, 2*self.C * self._sample_weight(s) / scipy.special.binom(self.n, s))
            weight = 1 / (prob * scipy.special.binom(self.n, s) * (self.n - s) * s )
            if self.paired_sampling and s == self.n // 2 and self.n % 2 == 0:
                # Partition the all middle sets into two
                # based on whether the combination contains n-1
                combo_gen = combination_generator(self.gen, self.n - 1, s-1, m_s)
                for indices in combo_gen:
                    self._add_one_sample(idx, list(indices) + [self.n-1], weight = weight)
                    idx += 1
            else:
                combo_gen = combination_generator(self.gen, self.n, s, m_s)
                for indices in combo_gen:
                    self.add_one_sample(idx, list(indices), weight = weight)
                    idx += 1

    def explain(self, explicand, num_samples):
        self.num_samples = int((num_samples - 2) // 2) * 2
        self.explicand = explicand

        if self.bernoulli_sampling:
            self._sample_without_replacement()
        else:
            self._sample_with_replacement()

        # A = Z P
        # y = v(z) - v0
        # b = y - Z1 (v1 - v0) / n    
        # (A^T S^T S A)^-1 A^T S^T S b + (v1 - v0) / n
        # (P^T Z^T S^T S Z P)^-1 P^T Z^T S^T S b + (v1 - v0) / n

        nonzero_mask = np.sum(self.SZ_binary, axis=1) != 0
        SZ_binary = self.SZ_binary[nonzero_mask]
        kernel_weights = np.array(self.kernel_weights)[nonzero_mask]

        v0 = self.model.predict(self.baseline)
        v1 = self.model.predict(self.explicand)
        inputs = self.baseline * (1 - SZ_binary) + self.explicand * SZ_binary
        Sy = self.model.predict(inputs) - v0

        SZ_binary1 = np.sum(SZ_binary, axis=1)
        Sb = Sy - (v1 - v0) * SZ_binary1 / self.n

        # Projection matrix
        P = np.eye(self.n) - (1.0 / self.n) * np.ones((self.n, self.n))

        PZSSb = P @ SZ_binary.T @ np.diag(kernel_weights) @ Sb
        PZSSZP = P @ SZ_binary.T @ np.diag(kernel_weights) @ SZ_binary @ P
        PZSSZP_inv_PZSSb = np.linalg.lstsq(PZSSZP, PZSSb, rcond=None)[0]

        phi = PZSSZP_inv_PZSSb + (v1 - v0) / self.n

        return phi
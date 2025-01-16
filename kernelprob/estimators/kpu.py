import numpy as np
import scipy
import scipy.special

from ..base_estimator import BaseEstimator
from ..p_generator import get_p
from .est_utils import combination_generator


class KernelProbUEstimator(BaseEstimator):
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
        self.p = get_p(baseline.shape[1], weighting)
        self.paired_sampling = paired_sampling
        self.leverage_sampling = leverage_sampling
        self.bernoulli_sampling = bernoulli_sampling

        # Number of features
        self.n = self.baseline.shape[1]

        # Random number generator
        self.gen = np.random.Generator(np.random.PCG64())

        # Compute constants (an, bn) based on p and n
        sum1, sum2 = 0, 0
        for ell in range(self.n):
            sum1 += scipy.special.binom(self.n - 1, ell) * self.p[ell] ** 2
            if ell >= 1:
                sum2 += scipy.special.binom(self.n - 2, ell - 1) * (self.p[ell] - self.p[ell - 1]) ** 2
        self.an = 2 * sum1 - sum2
        self.bn = sum2 / self.an

        self._sample_weight = lambda s: self._leverage_score(s) * scipy.special.binom(self.n, s)
        self._reweight = lambda s: 1 / self._sample_weight(s)

        # Will hold sample matrix (A_tilde) and kernel weights
        self.A_tilde = None
        self.kernel_weights = []

    def _leverage_score(self, s):
        first_term = 0  # |S| * p_{|S|-1}^2 + (n - |S|) * p_{|S|}^2
        second_term = 0 # (|S| p_{|S|-1} - (n-|S|) p_{|S|})^2
        if s > 0:
            first_term += s * self.p[s - 1] ** 2
            second_term += s * self.p[s - 1]
        if s < self.n:
            first_term += (self.n - s) * self.p[s] ** 2
            second_term -= (self.n - s) * self.p[s]

        # (first_term - (bn / (1 + n*bn)) * second_term^2) / an
        return (
            first_term
            - self.bn / (1 + self.n * self.bn) * (second_term ** 2)
        ) / self.an

    def _add_helper(self, idx, indices, weight):
        size = len(indices)
        indices_complement = np.array([i for i in range(self.n) if i not in indices])
        if size > 0:
            self.A_tilde[idx, indices] = self.p[size - 1]
        if size < self.n:
            self.A_tilde[idx, indices_complement] = -self.p[size]
        self.kernel_weights.append(weight)

    def _add_one_sample(self, idx, indices, weight):
        """
        Add one sampled set (and possibly its complement) 
        to self.A_tilde and self.kernel_weights.
        """
        if not self.paired_sampling:
            self._add_helper(idx, indices, weight)
        else:
            # For paired sampling, add set S at row 2*idx
            # and complement at row 2*idx + 1
            self._add_helper(2 * idx, indices, weight)
            indices_complement = np.array([i for i in range(self.n) if i not in indices])
            self._add_helper(2 * idx + 1, indices_complement, weight)

    def _find_constant_for_bernoulli(self, max_C=1e10):
        """
        Binary search for constant C so that the expected number
        of sampled sets matches self.num_samples (or 2^n).
        """
        m = min(self.num_samples, 2 ** self.n)  # Maximum number of samples is 2^n
         # Choose C so that sampling without replacement from min(1, C*prob) gives the same expected number of samples
        def expected_samples(C):
            return np.sum([min(scipy.special.binom(self.n, s), 2 * C * self._sample_weight(s))
                    for s in range(self.n + 1)
                    ])
        # Efficiently find C with binary search
        L = 1
        # Compute smallest probability
        R = 1 / min(self._sample_weight(s) for s in range(1, self.n + 1)) * 2 ** self.n
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
            # Avoid infinite loops:
            if abs(C_new - C) < 1e-12:
                break
            C = C_new

        self.C = round(C)

    def _sample_with_replacement(self):
        self.A_tilde = np.zeros((self.num_samples, self.n))
        valid_sizes = np.arange(self.n + 1)
        prob_sizes = np.array([self._sample_weight(s) for s in valid_sizes])
        prob_sizes /= prob_sizes.sum()
        # If paired_sampling, each "draw" adds 2 rows.
        num_draws = (
            self.num_samples if not self.paired_sampling else self.num_samples // 2
        )
        sampled_sizes = self.gen.choice(valid_sizes, num_draws, p=prob_sizes)

        for idx, s in enumerate(sampled_sizes):
            indices = self.gen.choice(self.n, s, replace=False)
            # weight = Pr(sampling this set) * w(s)
            weight = 1 / (self._sample_weight(s) * s * (self.n - s))
            self._add_one_sample(idx, indices, weight=weight)

    def _sample_without_replacement(self):
        self._find_constant_for_bernoulli()
        m_s_all = []
        for s in range(self.n):
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
        self.A_tilde = np.zeros((num_rows, self.n))
        idx = 0
        for s, m_s in enumerate(m_s_all):
            prob = min(1, 2*self.C * self._sample_weight(s) / scipy.special.binom(self.n, s))
            weight = 1 / prob
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
        self.explicand = explicand
        self.num_samples = int(num_samples // 2) * 2

        if self.bernoulli_sampling:
            self._sample_without_replacement()
        else:
            self._sample_with_replacement()

        # A_tilde = self.A_tilde / self.an
        A_tilde = self.A_tilde / self.an

        # Build binary matrix for each row: 1 if A_tilde>0, else 0
        binary_tilde = (A_tilde > 0).astype(int)

        # Evaluate model
        inputs = self.baseline * (1 - binary_tilde) + self.explicand * binary_tilde
        Sv = self.model.predict(inputs)

        # Weighted least squares
        ASSv = A_tilde.T @ np.diag(self.kernel_weights) @ Sv
        ASSA = A_tilde.T @ np.diag(self.kernel_weights) @ A_tilde

        x_star = np.linalg.lstsq(ASSA, ASSv, rcond=None)[0]

        # ATA = I + bn * 1 1^T
        ATA = np.eye(self.n) + self.bn * np.ones((self.n, self.n))
        phi = ATA @ x_star

        return phi
import math
from typing import Optional

import numpy as np

from ..base_estimator import BaseEstimator
from ..utils.p_generator import get_p, is_symmetric


def _harmonic_number(k: int) -> float:
    """H_k = 1 + 1/2 + … + 1/k."""
    return sum(1.0 / i for i in range(1, k + 1)) if k >= 1 else 0.0


class _GELSBase(BaseEstimator):
    """
    Parent class providing:

      * paired-sampling switch (variance-reduction when weights are symmetric);
      * _utility()  – creates hybrid instance;
      * _update_mean() – numerically stable online mean.
    """

    def __init__(
        self,
        model,
        baseline: np.ndarray,
        weighting: str = "shapley",
        *,
        paired: bool = True,
        random_state: Optional[int] = None,
    ):
        super().__init__(model, baseline, weighting)
        self.rng = np.random.default_rng(random_state)
        self.paired = paired

    def _utility(self, explicand: np.ndarray, active) -> float:
        x_base = np.asarray(self.baseline).ravel().copy()
        x_exp = np.asarray(explicand).ravel()

        idx = np.asarray(active, dtype=int)
        if idx.size:
            x_base[idx] = x_exp[idx]

        # model predict returns (1,…) – flatten to scalar
        return float(np.ravel(self.model.predict(x_base.reshape(1, -1)))[0])

    @staticmethod
    def _update_mean(curr: float, count: int, new_val: float) -> float:
        return curr + (new_val - curr) / (count + 1)


# GELS-R  (Algorithm 1) – ranking only 
# Yurong: not used in our paper
class GELSRankingEstimator(_GELSBase):
    """Generic Estimator based on Least Squares – ranking version."""

    def explain(self, explicand: np.ndarray, num_samples: int = 2048) -> np.ndarray:
        num_samples = num_samples // 2 if self.paired else num_samples
        n = explicand.size
        if n <= 1:
            return np.zeros(n)

        p = get_p(n, self.weighting)
        self.paired = is_symmetric(p)

        m = p[:-1] + p[1:]                           # m_s = p_s + p_{s+1}
        q_unscaled = np.array([math.comb(n, s) * m_s for s, m_s in enumerate(m, 1)])
        q_prob = q_unscaled / q_unscaled.sum()

        R = np.zeros(n)
        counts = np.zeros(n, dtype=int)
        all_idx = np.arange(n)

        for _ in range(num_samples):
            # sample a subset
            s = self.rng.choice(np.arange(1, n), p=q_prob)
            S = self.rng.choice(n, size=s, replace=False)
            u = self._utility(explicand, S)
            for i in S:
                R[i] = self._update_mean(R[i], counts[i], u)
                counts[i] += 1

            # complement sample (same weight) – does NOT advance loop counter
            if self.paired:
                S_bar = np.setdiff1d(all_idx, S, assume_unique=True)
                if S_bar.size:
                    u_bar = self._utility(explicand, S_bar)
                    for i in S_bar:
                        R[i] = self._update_mean(R[i], counts[i], u_bar)
                        counts[i] += 1

        return R  # ∝ ϕ + const·1 (ranking only)


# GELS (Algorithm 2) – full probabilistic value
class GELSEstimator(_GELSBase):
    """Generic Estimator based on Least Squares – full value."""

    def explain(self, explicand: np.ndarray, num_samples: int = 2048) -> np.ndarray:
        num_samples = num_samples // 2 if self.paired else num_samples
        n = explicand.size
        if n == 0:
            return np.array([])

        p = get_p(n, self.weighting)
        self.paired = is_symmetric(p)
        q_unscaled = np.array([math.comb(n + 1, s) * p_s for s, p_s in enumerate(p, 1)])
        q_prob = q_unscaled / q_unscaled.sum()

        R = np.zeros(n + 1)          # include null feature n
        counts = np.zeros(n + 1, dtype=int)
        all_idx = np.arange(n + 1)

        for _ in range(num_samples):
            s = self.rng.choice(np.arange(1, n + 1), p=q_prob)
            S = self.rng.choice(n + 1, size=s, replace=False)

            u = self._utility(explicand, [i for i in S if i < n])
            for i in S:
                R[i] = self._update_mean(R[i], counts[i], u)
                counts[i] += 1

            if self.paired:
                S_bar = np.setdiff1d(all_idx, S, assume_unique=True)
                u_bar = self._utility(explicand, [i for i in S_bar if i < n])
                for i in S_bar:
                    R[i] = self._update_mean(R[i], counts[i], u_bar)
                    counts[i] += 1

        C = sum((s / (n + 1)) * q_s for s, q_s in enumerate(q_unscaled, 1))
        return C * (R[:n] - R[n])    # drop null feature


# GELS-Shapley (Algorithm 3) – fast Shapley-specific
class GELSShapleyEstimator(_GELSBase):
    """Fast estimator specialised for the Shapley value."""

    def explain(self, explicand: np.ndarray, num_samples: int = 2048) -> np.ndarray:
        num_samples = num_samples // 2
        n = explicand.size
        if n < 2:
            return np.zeros_like(explicand, dtype=float)

        q_unscaled = np.array([n / (s * (n - s)) for s in range(1, n)])
        q_prob = q_unscaled / q_unscaled.sum()

        R = np.zeros(n)
        counts = np.zeros(n, dtype=int)
        all_idx = np.arange(n)

        for _ in range(num_samples):
            s = self.rng.choice(np.arange(1, n), p=q_prob)
            S = self.rng.choice(n, size=s, replace=False)

            u = self._utility(explicand, S)
            for i in S:
                R[i] = self._update_mean(R[i], counts[i], u)
                counts[i] += 1

            # if self.paired:
            S_bar = np.setdiff1d(all_idx, S, assume_unique=True)
            u_bar = self._utility(explicand, S_bar)
            for i in S_bar:
                R[i] = self._update_mean(R[i], counts[i], u_bar)
                counts[i] += 1

        R *= _harmonic_number(n - 1)
        offset = (
            self._utility(explicand, range(n))
            - self._utility(explicand, [])
            - R.sum()
        ) / n
        return R + offset

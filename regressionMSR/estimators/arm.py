import math
from typing import Optional, Sequence
import numpy as np

from ..base_estimator import BaseEstimator
from ..utils.p_generator import get_p, is_symmetric


class _ARMBase(BaseEstimator):

    def __init__(
        self,
        model,
        baseline: np.ndarray,
        weighting: str = "shapley",
        *,
        random_state: Optional[int] = None,
    ):
        super().__init__(model, baseline, weighting)
        self.rng = np.random.default_rng(random_state)

    def _utility(self, explicand: np.ndarray, active: Sequence[int]) -> float:
        exp_vec = np.asarray(explicand).ravel()
        base_vec = np.asarray(self.baseline).ravel()

        x = base_vec.copy()
        idx = np.asarray(active, dtype=int)
        if idx.size:
            x[idx] = exp_vec[idx]

        return float(np.ravel(self.model.predict(x.reshape(1, -1)))[0])

    @staticmethod
    def _update_mean(curr_mean: float, count: int, new_val: float) -> float:
        """
        Numerically stable incremental mean.
        """
        return curr_mean + (new_val - curr_mean) / (count + 1)


class ARMEstimator(_ARMBase):
    """
    Approximation-without-requiring-marginal (ARM) estimator
    for arbitrary probabilistic values.

    For each sample we either draw from P⁺ or P⁻
    (alternating if *paired* is True) and update *all* players
    whose expectation the sample contributes to.
    """

    def explain(self, explicand: np.ndarray, num_samples: int = 2048) -> np.ndarray:
        n = explicand.size
        if n == 0:
            return np.array([], dtype=float)

        p = get_p(n, self.weighting)
        self.paired = is_symmetric(p)

        # size probabilities  P⁺(|S| = s)  for 1 ≤ s ≤ n
        w_plus_sizes = np.array([math.comb(n, s) * p[s - 1] for s in range(1, n + 1)],
                                dtype=float)
        prob_plus_sizes = w_plus_sizes / w_plus_sizes.sum()

        # size probabilities  P⁻(|S| = s)  for 0 ≤ s ≤ n−1     (uses p_{s+1})
        w_minus_sizes = np.array([math.comb(n, s) * p[s] for s in range(0, n)],
                                 dtype=float)
        prob_minus_sizes = w_minus_sizes / w_minus_sizes.sum()


        pos_est = np.zeros(n, dtype=float)
        neg_est = np.zeros(n, dtype=float)
        pos_cnt = np.zeros(n, dtype=int)
        neg_cnt = np.zeros(n, dtype=int)

        all_idx = np.arange(n)


        next_is_plus = True  # used only if self.paired == True
        for _ in range(num_samples):
            if self.paired:
                draw_plus = next_is_plus
                next_is_plus = not next_is_plus
            else:
                # unbiased 50-50 choice
                draw_plus = (self.rng.random() < 0.5)

            if draw_plus:
                # P⁺ sample
                s = self.rng.choice(np.arange(1, n + 1), p=prob_plus_sizes)
                S = self.rng.choice(n, size=s, replace=False)
                u_val = self._utility(explicand, S)

                for i in S:
                    pos_est[i] = self._update_mean(pos_est[i], pos_cnt[i], u_val)
                    pos_cnt[i] += 1
            else:
                # P⁻ sample
                s = self.rng.choice(np.arange(0, n), p=prob_minus_sizes)
                S = self.rng.choice(n, size=s, replace=False)
                u_val = self._utility(explicand, S)

                # players *not* in S are updated
                not_S = np.setdiff1d(all_idx, S, assume_unique=True)
                for i in not_S:
                    neg_est[i] = self._update_mean(neg_est[i], neg_cnt[i], u_val)
                    neg_cnt[i] += 1

        return pos_est - neg_est

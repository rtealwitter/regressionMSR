import numpy as np
import math
from typing import Callable, Tuple

from ..base_estimator import BaseEstimator
from ..p_generator import get_p

class OFAEstimator(BaseEstimator):
    """
    OFA-optimal-paired (OFA-A) / OFA-fixed-paired (OFA-S) estimator.
    """
    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        baseline: np.ndarray,
        weighting: str = "shapley",
        mode: str="A", # "A" => OFA-A (optimal), "S" => OFA-S (fixed)
    ):
        super().__init__(
            model=model,
            baseline=baseline,
            weighting=weighting,
            )
        self.mode = mode

    def explain(
            self,
            explicand: np.ndarray,
            num_samples: int
    ) -> np.ndarray:
        """
        Returns
        -------
        phi : np.ndarray of shape (num_features,)
        """
        num_features = explicand.shape[1]

        p = get_p(num_features, self.weighting) # np.ndarray of shape (num_features,) - semivalue weights
        if p.shape[0] != num_features:
            raise ValueError(
                f"Expected p.shape[0] == num_features ({num_features}), got {p.shape[0]}"
            )
        # Multiply by binom(n-1, k-1) => semivalue weights
        p = p * np.array([
            math.comb(num_features - 1, i) for i in range(num_features)
        ], dtype=np.float64)
    
        if num_features < 4:
            raise ValueError("OFA expects num_features >= 4.")

        # Evaluate helper
        def evaluate(mask):
            x = np.where(mask, explicand, self.baseline)
            return self.model.predict(x.reshape(1, -1))

        # Precompute constants
        subset_mask = np.zeros(num_features, dtype=bool)

        v_empty = evaluate(subset_mask)
        v_singleton = np.empty(num_features, dtype=float)
        for i in range(num_features):
            subset_mask[i] = True
            v_singleton[i] = evaluate(subset_mask)
            subset_mask[i] = False

        subset_mask[:] = True
        v_full = evaluate(subset_mask)

        v_remove = np.empty(num_features, dtype=float)
        for i in range(num_features):
            subset_mask[i] = False
            v_remove[i] = evaluate(subset_mask)
            subset_mask[i] = True

        constants = (v_empty, v_full, v_singleton, v_remove)

        # Aggregation arrays
        results_aggregate = {
            "estimates": np.zeros((num_features, num_features - 3, 2), dtype=float),
            "counts":    np.zeros((num_features, num_features - 3, 2), dtype=int),
        }
        s_range = np.arange(2, num_features - 1)

        # Determine sampling distribution
        rng = np.random.default_rng()
        if self.mode == "A":
            tmp = (num_features / s_range)
            tmp = np.sqrt(tmp + tmp[::-1])
            p_sampling = tmp / tmp.sum()
        elif self.mode == "S":
            # "OFA_S" (fixed) for given weights `p`
            tmp = p[1:num_features - 2]**2 / s_range
            tmp += p[2:num_features - 1]**2 / np.arange(num_features - 2, 1, -1)
            tmp = tmp**0.5
            p_sampling = tmp / tmp.sum()
        else:
            raise ValueError("mode must be 'A' or 'S'.")

        # Paired sampling
        lock_switch = False
        prev_subset = None

        for _ in range(num_samples):
            if not lock_switch:
                # Normal draw
                s = rng.choice(s_range, p=p_sampling)
                chosen = rng.choice(num_features, size=s, replace=False)
                subset_bool = np.zeros(num_features, dtype=bool)
                subset_bool[chosen] = True
                prev_subset = subset_bool
                lock_switch = True
            else:
                # Inverse => complement
                if prev_subset is None:
                    raise RuntimeError("Paired logic error: no previous subset to invert.")
                subset_bool = ~prev_subset
                lock_switch = False

            v_val = evaluate(subset_bool)

            # Update aggregator
            idx = subset_bool.sum() - 2
            if 0 <= idx < (num_features - 3):
                # aggregator for subset
                counts_pre = results_aggregate["counts"][subset_bool, idx, 0]
                counts_cur = counts_pre + 1
                results_aggregate["estimates"][subset_bool, idx, 0] *= counts_pre / counts_cur
                results_aggregate["estimates"][subset_bool, idx, 0] += v_val / counts_cur
                results_aggregate["counts"][subset_bool, idx, 0] += 1

                # aggregator for complement
                subset_c = ~subset_bool
                counts_pre = results_aggregate["counts"][subset_c, idx, 1]
                counts_cur = counts_pre + 1
                results_aggregate["estimates"][subset_c, idx, 1] *= counts_pre / counts_cur
                results_aggregate["estimates"][subset_c, idx, 1] += v_val / counts_cur
                results_aggregate["counts"][subset_c, idx, 1] += 1

        # Final attribution
        v_empty, v_full, v_singleton, v_remove = constants
        est0 = results_aggregate["estimates"][:, :, 0]  # shape => (num_features, num_features-3)
        est1 = results_aggregate["estimates"][:, :, 1]

        tmp_val = (est0 * p[None, 1 : (num_features - 2)]).sum(axis=1)
        tmp_val += v_full * p[-1]
        tmp_val += v_singleton * p[0]
        tmp_val += (v_remove.sum() - v_remove) * p[-2] / (num_features - 1)

        tmp_val -= (est1 * p[None, 2 : (num_features - 1)]).sum(axis=1)
        tmp_val -= v_empty * p[0]
        tmp_val -= v_remove * p[-1]
        tmp_val -= (v_singleton.sum() - v_singleton) * p[1] / (num_features - 1)

        phi = tmp_val
        return phi
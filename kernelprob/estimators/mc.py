import numpy as np
from typing import Callable, Tuple
from ..base_estimator import BaseEstimator

class MonteCarloEstimator(BaseEstimator):
    """
    Generic Monte Carlo estimator that can use 
    Shapley-like or Banzhaf-like sampling logic.
    """
    
    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        baseline: np.ndarray,
        weighting: str = "shapley",
    ):
        """
        Generic Monte Carlo estimator
        """
        super().__init__(
            model=model,
            baseline=baseline,
            weighting=weighting,
            )
        if weighting not in {"shapley", "banzhaf"}:
            raise ValueError("weighting must be either 'shapley' or 'banzhaf'.")

        self._initialize_sampler()

    def explain(
            self,
            explicand: np.ndarray,
            num_samples: int,
    ) -> np.ndarray:
        """
        Compute the attributions using the Monte Carlo estimator.
        """
        n_features = self.baseline.shape[1]
        samples_per_feature = num_samples // n_features
        phi = np.zeros_like(self.baseline, dtype=float)

        for i in range(n_features):
            except_i = np.delete(np.arange(n_features), i)
            model_input = np.zeros((samples_per_feature * 2, n_features), dtype=float)
            sign = np.ones(samples_per_feature * 2, dtype=float)

            for S_idx in range(samples_per_feature):
                indices, indices_with_i = self.samplers[self.weighting](i, except_i)
                model_input[2 * S_idx, indices] = 1
                model_input[2 * S_idx + 1, indices_with_i] = 1
                sign[2 * S_idx] = -1  # Assign -1 for the without_i subset

            # Combine baseline and explicand
            model_input = self.baseline * (1 - model_input) + explicand * model_input

            # Model prediction
            model_output = self.model(model_input)

            # Aggregate attributions
            phi[:, i] = np.sum(model_output * sign) / samples_per_feature

        return phi

    def _initialize_sampler(self):
        self.rng = np.random.default_rng()
        self.samplers = {
            "shapley": self._mc_shapley_sampler,
            "banzhaf": self._mc_banzhaf_sampler
        }

    def _mc_shapley_sampler(self, i: int, except_i: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sampler for Shapley values.
        """
        size = self.rng.integers(0, len(except_i) + 1)
        indices = self.rng.choice(except_i, size=size, replace=False)
        indices_with_i = np.append(indices, i)
        return indices, indices_with_i

    def _mc_banzhaf_sampler(self, i: int, except_i: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sampler for Banzhaf values.
        """
        subset_mask = self.rng.integers(0, 2, size=len(except_i))
        indices = except_i[subset_mask == 1]
        indices_with_i = np.append(indices, i)
        return indices, indices_with_i

    
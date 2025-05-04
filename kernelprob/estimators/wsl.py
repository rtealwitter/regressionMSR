import math
import numpy as np

from ..base_estimator import BaseEstimator
from ..utils.p_generator import get_p


class WSLEstimator(BaseEstimator):
    """
    A unified Weighted Semivalue Lift (WSL)-style estimator that handles
    various weighting schemes (shapley, banzhaf, beta_shapley, weighted_banzhaf).
    
    Paired sampling is used: each 'round' includes a subset S, S + i, 
    complement(S), complement(S) + i, to reduce variance. If the total number 
    of draws is odd, we do one final 'round' (S, S + i) without complements.
    """

    def __init__(self, model, baseline: np.ndarray, weighting: str):
        super().__init__(model, baseline, weighting)

        # Number of features (players)
        self.num_player = self.baseline.shape[1] if self.baseline.ndim > 1 else self.baseline.size

        # Precompute semivalue weights for each possible subset size,
        # multiplied by the number of players (the typical WSL approach).
        p = get_p(self.num_player, weighting) * self.num_player
        self.weights = p * np.array(
            [math.comb(self.num_player - 1, i) for i in range(self.num_player)],
            dtype=np.float64,
        )

    def explain(self, explicand: np.ndarray, num_samples: int) -> np.ndarray:
        n_features = self.baseline.shape[1]
        num_samples = num_samples // 2
        self.nue_avg = num_samples // n_features

        self.rng = np.random.default_rng()
        phi = np.zeros_like(self.baseline, dtype=float)

        for i in range(n_features):
            except_i = np.delete(np.arange(n_features), i)
            model_input = np.zeros((self.nue_avg * 2, n_features), dtype=float)
            sign = np.ones(self.nue_avg * 2, dtype=float)
            weights = np.ones((self.nue_avg * 2), dtype=float)

            for S_idx in range(self.nue_avg):
                size = self.rng.integers(0, len(except_i) + 1)
                indices = self.rng.choice(except_i, size=size, replace=False)
                indices_with_i = np.append(indices, i)

                model_input[2 * S_idx, indices] = 1
                model_input[2 * S_idx + 1, indices_with_i] = 1

                sign[2 * S_idx] = -1  # Assign -1 for the without_i subset

                weights[2 * S_idx] = self.weights[len(indices)]
                weights[2 * S_idx + 1] = self.weights[len(indices)]

            model_input = self.baseline * (1 - model_input) + explicand * model_input
            model_output = self.model.predict(model_input)

            phi[:, i] = np.sum(model_output * sign * weights, axis=0) / self.nue_avg
        
        return phi
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

    # def explain(self, explicand: np.ndarray, num_samples: int) -> np.ndarray:
    #     n_features = self.num_player
    #     num_samples = num_samples // 2
    #     self.nue_avg = num_samples // n_features
    #     self.num_sample_pairs = self.nue_avg // 2
    #     leftover_flag = (self.nue_avg % 2)

    #     phi = np.zeros_like(self.baseline, dtype=float)
    #     rng = np.random.default_rng()

    #     total_rows = 4 * self.num_sample_pairs + 2 * leftover_flag

    #     for i in range(n_features):
    #         model_input = np.zeros((total_rows, n_features), dtype=float)
    #         sign_array = np.ones(total_rows, dtype=float)
    #         w_array = np.ones(total_rows, dtype=float)

    #         except_i = np.delete(np.arange(n_features), i)

    #         for pair_idx in range(self.num_sample_pairs):
    #             row_base = 4 * pair_idx

    #             size_s = rng.integers(0, len(except_i) + 1)

    #             indices_s = rng.choice(except_i, size=size_s, replace=False)
    #             indices_c = np.setdiff1d(except_i, indices_s)
    #             indices_s_i = np.append(indices_s, i)
    #             indices_c_i = np.append(indices_c, i)

    #             model_input[row_base + 0, indices_s] = 1.0
    #             model_input[row_base + 1, indices_s_i] = 1.0
    #             model_input[row_base + 2, indices_c] = 1.0
    #             model_input[row_base + 3, indices_c_i] = 1.0

    #             sign_array[row_base + 0] = -1.0
    #             sign_array[row_base + 2] = -1.0

    #             w_array[row_base + 0] = self.weights[size_s]
    #             w_array[row_base + 1] = self.weights[size_s]
    #             w_array[row_base + 2] = self.weights[len(indices_c)]
    #             w_array[row_base + 3] = self.weights[len(indices_c)]

    #         if leftover_flag == 1:
    #             row_base = 4 * self.num_sample_pairs
    #             size_s = rng.integers(0, len(except_i) + 1)
    #             indices_s = rng.choice(except_i, size=size_s, replace=False)
    #             indices_s_i = np.append(indices_s, i)

    #             model_input[row_base + 0, indices_s] = 1.0
    #             model_input[row_base + 1, indices_s_i] = 1.0

    #             sign_array[row_base + 0] = -1.0
    #             sign_array[row_base + 1] = +1.0

    #             w_array[row_base + 0] = self.weights[size_s]
    #             w_array[row_base + 1] = self.weights[size_s]

    #         input_matrix = (
    #             self.baseline * (1.0 - model_input) + explicand * model_input
    #         )
    #         model_output = self.model.predict(input_matrix)  # shape: (total_rows,) or (total_rows,1)
    #         phi[:, i] = np.sum(model_output * sign_array * w_array, axis=0) / self.nue_avg

    #     return phi

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
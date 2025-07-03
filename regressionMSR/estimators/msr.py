import numpy as np
from typing import Callable
from ..base_estimator import BaseEstimator

class MSREstimator(BaseEstimator):
    """
    MSR estimator only for Banzhaf values.
    """
    def __init__(
        self,
        model: Callable[[np.ndarray], np.ndarray],
        baseline: np.ndarray,
        weighting: str = "shapley",
    ):
        super().__init__(
            model=model,
            baseline=baseline,
            weighting=weighting,
            )
        if weighting not in {"banzhaf"}:
            raise ValueError("weighting must be 'banzhaf'.")
    
    def explain(
            self,
            explicand: np.ndarray,
            num_samples: int,
    ) -> np.ndarray:
        
        n = self.baseline.shape[1]
        phi = np.zeros_like(self.baseline)
        gen = np.random.Generator(np.random.PCG64())

        subset_masks = gen.integers(0, 2, size=(num_samples, n))
        model_input = np.zeros((num_samples, n))
        model_input = self.baseline * (1-subset_masks) + explicand * subset_masks
        model_output = self.model.predict(model_input)

        for i in range(n):
            with_i = np.where(subset_masks[:, i] == 1)
            without_i = np.where(subset_masks[:, i] == 0)
            if with_i[0].shape[0] == 0 or without_i[0].shape[0] == 0:
                phi[:, i] = 0
                print(f"Feature {i} is not in any subset")
            else:
                phi[:, i] = np.mean(model_output[with_i]) - np.mean(model_output[without_i])

        return phi
import numpy as np
from sklearn.linear_model import LinearRegression

from ..base_estimator import BaseEstimator

class KernelBanzhafEstimator(BaseEstimator):
    """
    Kernel Banzhaf method: regression approach (similar to Kernel SHAP but 
    with Banzhaf weighting).
    """
    def __init__(
        self,
        model,
        baseline,
        weighting="banzhaf"
    ):
        super().__init__(
            model=model,
            baseline=baseline,
            weighting=weighting,
            )

    def explain(self, explicand, num_samples):
        n = self.baseline.shape[1]
        weight = 1/2

        phi = np.zeros_like(self.baseline)
        gen = np.random.Generator(np.random.PCG64())

        masks_original = gen.integers(0, 2, size=(num_samples // 2, n))
        masks_complement = 1 - masks_original
        masks = np.vstack((masks_original, masks_complement))

        model_input = self.baseline * (1 - masks) + explicand * masks

        lr_features = masks - weight
        lr_output = self.model.predict(model_input)

        l_model = LinearRegression().fit(lr_features, lr_output)
        values = l_model.coef_.T

        for i in range(n):
            phi[:, i] = values[i]

        return phi
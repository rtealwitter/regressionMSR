# TODO: implement shap_iq

from ..base_estimator import BaseEstimator

class SHAPIQEstimator(BaseEstimator):
    def __init__(self, model, baseline, weighting):
        super().__init__(model, baseline, weighting)

    def explain(self, explicand, num_samples):
        return super().explain(explicand, num_samples)
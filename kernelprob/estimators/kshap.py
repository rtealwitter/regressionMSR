import shap

from ..base_estimator import BaseEstimator

class KernelSHAPEstimator(BaseEstimator):
    def __init__(self, model, baseline, weighting):
        super().__init__(model, baseline, weighting)
        self.model = model
        self.baseline = baseline

    def explain(self, explicand, num_samples):
        eval_model = lambda X : self.model.predict(X)
        explainer = shap.KernelExplainer(eval_model, self.baseline)
        shap_values = explainer.shap_values(explicand, nsamples=num_samples, silent=True, l1_reg=False)
        return shap_values
import shap

from ..base_estimator import BaseEstimator

class PermutationSHAPEstimator(BaseEstimator):
    def __init__(self, model, baseline, weighting):
        super().__init__(model, baseline, weighting)
        self.model = model
        self.baseline = baseline

    def explain(self, explicand, num_samples):
        eval_model = lambda X : self.model.predict(X)
        explicand = explicand.astype('float64')
        num_features = explicand.shape[1]
        num_permutations = num_samples // num_features

        explainer = shap.PermutationExplainer(eval_model, self.baseline)
        shap_values = explainer.shap_values(explicand, npermutations=num_permutations, silent=True)
        return shap_values
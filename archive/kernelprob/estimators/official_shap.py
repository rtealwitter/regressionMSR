import shap
import numpy as np

def official_kernel_shap(baseline, explicand, model, num_samples):
    eval_model = lambda X : model.predict(X)

    explainer = shap.KernelExplainer(eval_model, baseline)
    shap_values = explainer.shap_values(explicand, nsamples=num_samples, silent=True, l1_reg=False)
    return shap_values

def official_permutation_shap(baseline, explicand, model, num_samples):
    eval_model = lambda X : model.predict(X)
    explicand = explicand.astype('float64')
    num_features = explicand.shape[1]
    num_permutations = num_samples // num_features

    explainer = shap.PermutationExplainer(eval_model, baseline)
    shap_values = explainer.shap_values(explicand, npermutations=num_permutations, silent=True)
    return shap_values

def official_shapley_sampling(baseline, explicand, model, num_samples):
    eval_model = lambda X : model.predict(X)
    explainer = shap.SamplingExplainer(eval_model, baseline)
    shap_values = explainer.shap_values(explicand, nsamples=num_samples, silent=True)
    return shap_values

def official_tree_shap(baseline, explicand, model, num_samples):
    # Suppress warning only for this function
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    explainer = shap.TreeExplainer(model, baseline)
    shap_values = explainer.shap_values(explicand)
    # Re-enable warnings
    warnings.filterwarnings("default", category=UserWarning)
    return shap_values
import shap

def tree_shap(baseline, explicands, model):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    explainer = shap.TreeExplainer(model, data=baseline)
    shap_values = []
    for explicand in explicands:
        shap_values.append(explainer.shap_values(explicand))
        
    # Re-enable warnings
    warnings.filterwarnings("default", category=UserWarning)
    return shap_values
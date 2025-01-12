import time
import sklearn.ensemble
import xgboost
import numpy as np
import shap

from treeshap_path import TreeExplainer
from treeshap_int import TreeInterventionalExplainer
from exact_enumerate import shapley_enumerate_all_subsets, banzhaf_enumerate_all_subsets, beta_shapley_enumerate_all_subsets
from treeprob import TreeProbExplainer, tree_prob

def official_tree_shap(baseline, explicand, model):
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    explainer = shap.TreeExplainer(model, baseline, feature_perturbation="interventional")
    shap_values = explainer.shap_values(explicand)
    warnings.filterwarnings("default", category=UserWarning)
    return shap_values

def load_input(X, seed=None):
    if seed is not None:
        np.random.seed(seed)
    baseline = X.mean().values.reshape(1, -1)
    explicand_idx = np.random.choice(X.shape[0])
    explicand = X.iloc[explicand_idx].values.reshape(1, -1)
    for i in range(explicand.shape[1]):
        while baseline[0, i] == explicand[0, i]:
            explicand_idx = np.random.choice(X.shape[0])
            explicand[0,i] = X.iloc[explicand_idx, i]
    return baseline, explicand


def main():
    random_state = 42

    X, y = shap.datasets.california(n_points=1000)
    # X, y = shap.datasets.communitiesandcrime()
    print(X.shape, y.shape)
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=4, random_state=random_state)
    model.fit(X, y)

    baseline, explicand = load_input(X, seed=random_state)

    phi = tree_prob(
        baseline=baseline,
        explicand=explicand,
        model=model,
        weighting="banzhaf"
    )

    print(f"TreeProb values: {phi}")

    # start_time = time.time()
    # true_shap_values = official_tree_shap(baseline, explicand, model).flatten()
    # print(f"Official Tree SHAP: {time.time() - start_time}")

    # start_time = time.time()
    # explainer = TreeProbExplainer(model)
    # phi = explainer.prob_values(
    #     X=explicand,
    #     refset=baseline,
    # )
    # print(f"TreeProbExplainer: {time.time() - start_time}")

    # print(f"True SHAP values: {true_shap_values}")
    # print(f"TreeProb SHAP values: {phi}")

    # int_explainer = TreeInterventionalExplainer(model)
    # int_shap_values = int_explainer.shap_values(
    #     X=explicand, 
    #     refset=baseline
    # )
    # print(f"TreeInterventional SHAP values: {int_shap_values}")


    # banzhaf values
    true_banzhaf = banzhaf_enumerate_all_subsets(model, explicand, baseline)
    print(f"True Banzhaf values: {true_banzhaf}")

    # beta shapley values
    # true_beta_shapley = beta_shapley_enumerate_all_subsets(model, explicand, baseline, alpha=1, beta=1)
    # print(f"True Beta Shapley values: {true_beta_shapley}")

if __name__ == "__main__":
    main()
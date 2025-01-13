import time
import sklearn.ensemble
import xgboost as xgb
import numpy as np
import shap

from treeprob import tree_prob


def official_tree_shap(baseline, explicand, model):
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    explainer = shap.TreeExplainer(
        model, baseline, feature_perturbation="interventional"
    )
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
            explicand[0, i] = X.iloc[explicand_idx, i]
    return baseline, explicand


def main():
    random_state = 42

    # X, y = shap.datasets.california(n_points=1000)
    X, y = shap.datasets.communitiesandcrime()
    print(X.shape, y.shape)
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=4, random_state=random_state)
    # model = xgb.XGBRegressor(n_estimators=100, max_depth=4, random_state=random_state)
    model.fit(X, y)
    baseline, explicand = load_input(X, seed=random_state)

    start_time = time.time()
    phi = tree_prob(
        baseline=baseline,
        explicand=explicand,
        model=model,
        weighting="shapley",
    )
    print(f"TreeProb: {time.time() - start_time}")

    start_time = time.time()
    true_shap_values = official_tree_shap(baseline, explicand, model).flatten()
    print(f"Official Tree SHAP: {time.time() - start_time}")

    print(f"TreeProb values: {phi}")
    print(f"True SHAP values: {true_shap_values}")


if __name__ == "__main__":
    main()

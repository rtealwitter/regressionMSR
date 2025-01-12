import time
import sklearn.ensemble
import xgboost
import numpy as np
import shap

from treeshap_path import TreeExplainer
from treeshap_int import TreeInterventionalExplainer
from exact_enumerate import shapley_enumerate_all_subsets

def official_tree_shap(baseline, explicand, model):
    # Suppress warning only for this function
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    explainer = shap.TreeExplainer(model, baseline, feature_perturbation="interventional")
    shap_values = explainer.shap_values(explicand)
    # Re-enable warnings
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

    # X, y = shap.datasets.california(n_points=1000)
    X, y = shap.datasets.communitiesandcrime()
    print(X.shape, y.shape)
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=4, random_state=random_state)
    model.fit(X, y)

    # estimator = model.estimators_[0]

    # # Accessing the attributes of the estimator
    # print("Number of nodes:", estimator.tree_.node_count)
    # print("Thresholds:", estimator.tree_.threshold)
    # print("Features:", estimator.tree_.feature)
    # print("Values:", estimator.tree_.value)
    # print("Children:", estimator.tree_.children_left, estimator.tree_.children_right)

    baseline, explicand = load_input(X, seed=random_state)

    # print(f'baseline: {baseline}')
    # print(f'explicand: {explicand}')

    # explainer = shap.TreeExplainer(model, feature_perturbation="path_dependent")
    # c_shap_values = explainer.shap_values(explicand)
    # py_shap_values = TreeExplainer(model).shap_values(explicand)

    # print(f'c_shap_values: {c_shap_values}')
    # print(f'py_shap_values: {py_shap_values}')


    start_time = time.time()
    true_shap_values = official_tree_shap(baseline, explicand, model).flatten()
    print(f"Official Tree SHAP: {time.time() - start_time}")

    int_explainer = TreeInterventionalExplainer(model)

    start_time = time.time()
    int_shap_values = int_explainer.shap_values(
        X=explicand, 
        refset=baseline
    )
    print(f"Interventional SHAP: {time.time() - start_time}")

    # start_time = time.time()
    # phi = shapley_enumerate_all_subsets(model, explicand, baseline)
    # print(f"Exact SHAP: {time.time() - start_time}")

    print(f'true_shap_values: {true_shap_values}')
    print(f'interventional_shap_values: {int_shap_values}')
    # print(f'shapley_values: {phi}')


if __name__ == "__main__":
    main()
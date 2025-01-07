import time
import sklearn.ensemble
import xgboost
import numpy as np
import shap

from treeshap_path import TreeExplainer
from treeshap_int import TreeInterventionalExplainer
from exact_shap import shapley_enumerate_all_subsets

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

    X, y = shap.datasets.california(n_points=1000)
    print(X.shape, y.shape)
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=4, random_state=random_state)
    model.fit(X, y)

    baseline, explicand = load_input(X, seed=random_state)

    start_time = time.time()
    true_shap_values = official_tree_shap(baseline, explicand, model).flatten()
    print(f"Official Tree SHAP: {time.time() - start_time}")

    # Xd = xgboost.DMatrix(X, label=y)
    # bst = xgboost.train({"learning_rate": 0.01, "max_depth": 4}, Xd, 1000)

    # start = time.time()
    # cpred = model.predict(X)
    # explainer = shap.TreeExplainer(model)
    # explanation = explainer(Xd)
    # shap_values = explanation.values
    # print(time.time() - start)

    # start = time.time()
    # py_shap_values = TreeExplainer(model).shap_values(X.iloc[:, :])
    # print(time.time() - start)

    # print(f'c_shap_values: {c_shap_values}')
    # print(f'py_shap_values: {py_shap_values}')


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
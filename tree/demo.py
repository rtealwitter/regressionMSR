import numpy as np
from sklearn.ensemble import RandomForestRegressor

from treeshap_int import TreeInterventionalExplainer
from main import official_tree_shap
from exact_shap import shapley_enumerate_all_subsets

# Suppose a trivial dataset
X = np.array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]

], dtype=float)
y = X[:,0] + 2*X[:,1] + 3*X[:,2]

model = RandomForestRegressor(n_estimators=1, max_depth=3, random_state=0)
model.fit(X, y)

# estimator = model.estimators_[0]

# # Accessing the attributes of the estimator
# print("Number of nodes:", estimator.tree_.node_count)
# print("Thresholds:", estimator.tree_.threshold)
# print("Features:", estimator.tree_.feature)
# print("Values (leaf nodes):", estimator.tree_.value)
# print("Children:", estimator.tree_.children_left, estimator.tree_.children_right)

# exit()

# Baseline: pick just [0,0] as baseline
baseline = np.array([[0,0,0]], dtype=float)

# Explicand: let's pick [1,1]
explicand = np.array([[1,1,1]], dtype=float)

# Use your two-path code
interventional_explainer = TreeInterventionalExplainer(model)
phi = interventional_explainer.shap_values(explicand, refset=baseline)
print("phi:", phi)

# Use official SHAP
true_shap_values = official_tree_shap(baseline, explicand, model).flatten()
print("true_shap_values:", true_shap_values)

# Use exact Shapley
exact_phi = shapley_enumerate_all_subsets(model, explicand, baseline)
print("exact_phi:", exact_phi)
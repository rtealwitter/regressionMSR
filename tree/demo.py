import numpy as np
from sklearn.ensemble import RandomForestRegressor

from treeshap_int import TreeInterventionalExplainer
from kernelprob.tree.main_shap import official_tree_shap
from kernelprob.tree.exact_enumerate import shapley_enumerate_all_subsets

# Suppose a trivial dataset
X = np.array([
    [0, 0, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 1, 1],
    [0, 1, 0, 0],
    [0, 1, 0, 1],
    [0, 1, 1, 0],
    [0, 1, 1, 1],
    [1, 0, 0, 0],
    [1, 0, 0, 1],
    [1, 0, 1, 0],
    [1, 0, 1, 1],
    [1, 1, 0, 0],
    [1, 1, 0, 1],
    [1, 1, 1, 0],
    [1, 1, 1, 1],
], dtype=float)
y = X[:,0] + 2*X[:,1] + 3*X[:,2] + 4*X[:,3]

model = RandomForestRegressor(n_estimators=5, max_depth=5, random_state=0)
model.fit(X, y)

# Baseline: pick just [0,0] as baseline
baseline = np.array([[0,0,0,1]], dtype=float)

# Explicand: let's pick [1,1]
explicand = np.array([[1,1,1,0]], dtype=float)

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
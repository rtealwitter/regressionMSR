import scipy.special
from .official_shap import *
from .regression import *
from .matrix import *
from .naive import *
from .treeprob.treeprob import *
from .regression_prob import *
from .p_generator import *

import numpy as np
import xgboost as xgb
import scipy

# # # Probabilistic estimators # # #
# Completed: Kernel Prob, Tree Prob
# TODO: One for all, Monte Carlo

prob_estimators = {
    'Kernel Prob': kernel_prob,
    'Tree Prob': tree_prob,
}

# # # SHAP estimators # # #
# Completed: Optimized Kernel SHAP, Leverage SHAP, Official Tree SHAP, Permutation SHAP
# Old: Monte Carlo, Matrix SHAP (+ variants), Leverage SHAP variants, 

shap_estimators = {
    'Optimized Kernel SHAP': official_kernel_shap,
    'Leverage SHAP': leverage_shap,
    'Official Tree SHAP': official_tree_shap,
    'Permutation SHAP': official_permutation_shap,
}

# # # Banzhaf estimators # # #
# TODO: Kernel Banzhaf, MSR
banzhaf_estimators = {}


# # # All estimators # # #
# Combine estimators
estimators = {}
for estimator_dic in [prob_estimators, shap_estimators]:
    for key, val in estimator_dic.items():
        estimators[key] = val

#estimators = {
#    'Kernel Prob': kernel_prob,
#    'Monte Carlo': monte_carlo,
#    'Matrix SHAP': matrix_shap,
#    'Matrix SHAP wo Bernoulli': matrix_shap_wo_bernoulli,
#    'Matrix SHAP wo Bernoulli, Paired': matrix_shap_wo_bernoulli_paired,
#    'Kernel SHAP': kernel_shap,
#    'Optimized Kernel SHAP': official_kernel_shap,
#    'Leverage SHAP': leverage_shap,
#    'Kernel SHAP Paired': kernel_shap_paired,
#    'Leverage SHAP wo Bernoulli': leverage_shap_wo_bernoulli,
#    'Leverage SHAP wo Bernoulli, Paired': leverage_shap_wo_bernoulli_paired,
#    'Official Tree SHAP': official_tree_shap,
#    'Permutation SHAP': official_permutation_shap,
#    'Tree Prob': tree_prob,
#}

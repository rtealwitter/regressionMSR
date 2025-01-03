import scipy.special
from .official_shap import *
from .regression import *
from .matrix import *
from .naive import *

import numpy as np
import xgboost as xgb
import scipy

estimators = {
    'Monte Carlo': monte_carlo,
    'Matrix SHAP': matrix_shap,
    'Matrix SHAP wo Bernoulli': matrix_shap_wo_bernoulli,
    'Matrix SHAP wo Bernoulli, Paired': matrix_shap_wo_bernoulli_paired,
    'Kernel SHAP': kernel_shap,
    'Optimized Kernel SHAP': official_kernel_shap,
    'Leverage SHAP': leverage_shap,
    'Kernel SHAP Paired': kernel_shap_paired,
    'Leverage SHAP wo Bernoulli': leverage_shap_wo_bernoulli,
    'Leverage SHAP wo Bernoulli, Paired': leverage_shap_wo_bernoulli_paired,
    'Official Tree SHAP': official_tree_shap,
    'Permutation SHAP': official_permutation_shap,
}

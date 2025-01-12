import itertools
import numpy as np
from math import factorial
import warnings

from weight_utils import factorial, beta_function

def shapley_enumerate_all_subsets(model, explicand, baseline):
    """
    Compute exact Shapley values for a single instance explicand by enumerating
    all subsets of features, using a baseline dataset for marginalizing
    out absent features.
    """
    warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

    explicand = np.array(explicand, dtype=float).ravel()
    M = explicand.shape[0]

    # Precompute factorial-based weights for Shapley
    # weight(|S|) = |S|! * (M - |S| - 1)! / M!
    fact = [factorial(i) for i in range(M+1)]  # factorial from 0..M
    denom = fact[M]  # M!
    
    def shapley_weight(subset_size):
        # subset_size = |S|
        return fact[subset_size] * fact[M - subset_size - 1] / denom

    def f_of_subset(subset):
        subset_mask = np.zeros(M, dtype=bool)
        subset_mask[list(subset)] = True

        preds = 0.0
        N = baseline.shape[0]
        for i in range(N):
            temp = baseline[i, :].copy()
            temp[subset_mask] = explicand[subset_mask]
            preds += model.predict(temp.reshape(1, -1))[0]
        return preds / N

    f_cache = dict()

    all_features = range(M)
    for r in range(M+1):  # r = subset size
        for subset in itertools.combinations(all_features, r):
            s_key = frozenset(subset)
            f_cache[s_key] = f_of_subset(subset)

    shap_values = np.zeros(M, dtype=float)

    # sum_{S not containing i} [ weight(|S|) * (f(S union i) - f(S)) ]
    for i in range(M):
        for r in range(M):  # subsets of size r
            for subset in itertools.combinations([f for f in all_features if f != i], r):
                s_key = frozenset(subset)
                s_union_i_key = frozenset(subset).union({i})
                subset_size = r  # i.e. |S|
                # contribution = [ f(S union i) - f(S) ] * shapley_weight(|S|)]
                contribution = (
                    f_cache[s_union_i_key] 
                    - f_cache[s_key]
                ) * shapley_weight(subset_size)
                shap_values[i] += contribution

    return shap_values


def banzhaf_enumerate_all_subsets(model, explicand, baseline):
    """
    Compute exact Banzhaf values for a single instance x by enumerating
    all subsets of features, using a baseline dataset for marginalizing
    out absent features.
    """
    warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

    explicand = np.array(explicand, dtype=float).ravel()
    M = explicand.shape[0]
    N = baseline.shape[0]

    # Helper function to compute f(S), i.e., model prediction
    # when only features in S come from x (others from baseline).
    def f_of_subset(subset):
        subset_mask = np.zeros(M, dtype=bool)
        subset_mask[list(subset)] = True

        preds = 0.0
        for i in range(N):
            temp = baseline[i, :].copy()
            temp[subset_mask] = explicand[subset_mask]
            preds += model.predict(temp.reshape(1, -1))[0]
        return preds / N

    # Precompute model outputs for all subsets
    f_cache = {}
    all_features = range(M)
    for r in range(M + 1):
        for subset in itertools.combinations(all_features, r):
            s_key = frozenset(subset)
            f_cache[s_key] = f_of_subset(subset)

    banzhaf_values = np.zeros(M, dtype=float)

    for i in range(M):
        contribution_sum = 0.0
        for r in range(M):
            for subset in itertools.combinations([f for f in all_features if f != i], r):
                s_key = frozenset(subset)
                s_union_i_key = s_key.union({i})
                contribution_sum += (f_cache[s_union_i_key] - f_cache[s_key])
        
        banzhaf_values[i] = contribution_sum / (2 ** (M - 1))

    return banzhaf_values


def weight_beta_shapley(M, subset_size, alpha, beta):
    # TODO: check beta shapley domain
    """
    Compute the Beta-weighted Shapley coefficient for a coalition of size 'subset_size'
    out of M total features, with hyperparameters alpha, beta.
    """
    numerator = beta_function(subset_size + beta - 1, (M - subset_size) + alpha)
    denominator = beta_function(alpha, beta)
    beta_factor = (numerator / denominator) * M
    if subset_size == 0:
        shap_term = factorial(M - 1) / (1 * factorial(M))  # effectively 1 / M
    else:
        shap_term = factorial(M - 1) / (factorial(subset_size - 1) * factorial(M - subset_size))
    
    return beta_factor * shap_term

def beta_shapley_enumerate_all_subsets(
    model, 
    explicand, 
    baseline, 
    alpha=1.0, 
    beta=1.0
):
    """
    Compute ground truth Beta Shapley values for a single instance ('explicand')
    by enumerating all subsets of features, using a baseline dataset to marginalize
    out absent features. This is an exponential-time method (O(2^M)) and is
    typically only feasible for a moderate number of features.
    """
    warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

    # Convert explicand to a flat NumPy array
    explicand = np.array(explicand, dtype=float).ravel()
    M = explicand.shape[0]
    N = baseline.shape[0]

    def f_of_subset(subset):
        subset_mask = np.zeros(M, dtype=bool)
        subset_mask[list(subset)] = True

        preds = 0.0
        for i in range(N):
            temp = baseline[i, :].copy()
            temp[subset_mask] = explicand[subset_mask]
            preds += model.predict(temp.reshape(1, -1))[0]
        return preds / N

    f_cache = {}
    all_features = range(M)
    for r in range(M + 1):
        for subset in itertools.combinations(all_features, r):
            s_key = frozenset(subset)
            f_cache[s_key] = f_of_subset(subset)

    beta_shap_values = np.zeros(M, dtype=float)
    
    # For each feature i, sum over subsets S (not containing i):
    #    Beta-weight(|S|) * [ f(S ∪ {i}) - f(S) ]
    for i in range(M):
        for r in range(M):  # subsets of size r
            for subset in itertools.combinations([f for f in all_features if f != i], r):
                s_key = frozenset(subset)
                s_union_i_key = s_key.union({i})

                # f(S union i) - f(S)
                marginal_contribution = f_cache[s_union_i_key] - f_cache[s_key]

                weight = weight_beta_shapley(M, r, alpha, beta)

                beta_shap_values[i] += marginal_contribution * weight

    return beta_shap_values
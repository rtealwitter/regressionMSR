import itertools
import numpy as np
from math import factorial
import warnings

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ..treeprob.weights import factorial, w_beta_shapley, w_weighted_banzhaf


def enumerate_all_subsets(baseline, explicand, model, weighting="banzhaf"):
    weighting_list = weighting.split('_')
    if "beta" in weighting:
        if len(weighting_list) == 2:
            alpha, beta = (1, 1)
        else:
            alpha_str, beta_str = weighting_list[-2], weighting_list[-1]
            alpha, beta = (float(alpha_str), float(beta_str))
        weighting = "beta_shapley"
    elif "weighted" in weighting:
        if len(weighting_list) == 2:
            p = 0.5
        else:
            p = float(weighting_list[-1])
        weighting = "weighted_banzhaf"
    if weighting == "shapley":
        return shapley_enumerate_all_subsets(model, explicand, baseline)
    elif weighting == "banzhaf":
        return banzhaf_enumerate_all_subsets(model, explicand, baseline)
    elif weighting == "beta_shapley":
        return beta_shapley_enumerate_all_subsets(model, explicand, baseline, alpha, beta)
    elif weighting == "weighted_banzhaf":
        return weighted_banzhaf_enumerate_all_subsets(model, explicand, baseline, p)
    else:
        raise ValueError(f"Unknown weighting method: {weighting}")

def shapley_enumerate_all_subsets(model, explicand, baseline):
    """
    Compute exact Shapley values for an explicand by enumerating
    all subsets of features, using a baseline dataset for marginalizing
    out absent features.
    """
    warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

    explicand = np.array(explicand, dtype=float).ravel()
    M = explicand.shape[0]

    # Precompute factorial-based weights for Shapley
    # weight(|S|) = |S|! * (M - |S| - 1)! / M!
    fact = [factorial(i) for i in range(M + 1)]  # factorial from 0..M
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
    for r in range(M + 1):  # r = subset size
        for subset in itertools.combinations(all_features, r):
            s_key = frozenset(subset)
            f_cache[s_key] = f_of_subset(subset)

    shap_values = np.zeros(M, dtype=float)

    # sum_{S not containing i} [ weight(|S|) * (f(S union i) - f(S)) ]
    for i in range(M):
        for r in range(M):  # subsets of size r
            for subset in itertools.combinations(
                [f for f in all_features if f != i], r
            ):
                s_key = frozenset(subset)
                s_union_i_key = frozenset(subset).union({i})
                subset_size = r  # i.e. |S|
                # contribution = [ f(S union i) - f(S) ] * shapley_weight(|S|)]
                contribution = (
                    f_cache[s_union_i_key] - f_cache[s_key]
                ) * shapley_weight(subset_size)
                shap_values[i] += contribution

    return shap_values


def banzhaf_enumerate_all_subsets(model, explicand, baseline):
    """
    Compute exact Banzhaf values for an explicand by enumerating
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
            for subset in itertools.combinations(
                [f for f in all_features if f != i], r
            ):
                s_key = frozenset(subset)
                s_union_i_key = s_key.union({i})
                contribution_sum += f_cache[s_union_i_key] - f_cache[s_key]

        banzhaf_values[i] = contribution_sum / (2 ** (M - 1))

    return banzhaf_values


def beta_shapley_enumerate_all_subsets(model, explicand, baseline, alpha=1.0, beta=1.0):
    """
    Compute ground truth Beta Shapley values for an explicand
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
            for subset in itertools.combinations(
                [f for f in all_features if f != i], r
            ):
                s_key = frozenset(subset)
                s_union_i_key = s_key.union({i})

                # f(S union i) - f(S)
                marginal_contribution = f_cache[s_union_i_key] - f_cache[s_key]

                weight = w_beta_shapley(M, r, alpha, beta)

                beta_shap_values[i] += marginal_contribution * weight

    return beta_shap_values


def weighted_banzhaf_enumerate_all_subsets(model, explicand, baseline, p=0.5):
    """
    Compute exact Weighted Banzhaf values for an explicand
    by enumerating all subsets of features, using a baseline dataset to marginalize
    out absent features. This is exponential in M and typically only feasible for
    moderately small M.
    """
    warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

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

    wb_values = np.zeros(M, dtype=float)

    for i in range(M):
        value_i = 0.0
        exclude_i = [f for f in all_features if f != i]
        for r in range(M):
            w_r = w_weighted_banzhaf(M, r, p)
            for subset in itertools.combinations(exclude_i, r):
                s_key = frozenset(subset)
                s_union_i_key = frozenset(subset).union({i})
                contrib = f_cache[s_union_i_key] - f_cache[s_key]
                value_i += w_r * contrib

        wb_values[i] = value_i

    return wb_values

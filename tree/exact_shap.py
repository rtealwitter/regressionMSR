import itertools
import numpy as np
from math import factorial
import warnings

def shapley_enumerate_all_subsets(model, x, background):
    """
    Compute exact Shapley values for a single instance x by enumerating
    all subsets of features, using a background dataset for marginalizing
    out absent features.

    Parameters
    ----------
    model : Any regressor with a 'predict' method in scikit-learn style.
    x : numpy.ndarray of shape (M,)
        The single instance we want to explain.
    background : numpy.ndarray of shape (N, M)
        A dataset representing the distribution for 'absent' features.
        For each subset S, the absent features are taken from these N
        background samples, and the model outputs are averaged.

    Returns
    -------
    shap_values : numpy.ndarray of shape (M,)
        The exact Shapley values for instance x.
    """
    warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

    x = np.array(x, dtype=float).ravel()
    M = x.shape[0]

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
        N = background.shape[0]
        for i in range(N):
            # Make a copy of the i-th background row
            temp = background[i, :].copy()
            # Overwrite features in subset with x
            temp[subset_mask] = x[subset_mask]
            preds += model.predict(temp.reshape(1, -1))[0]
        return preds / N

    f_cache = dict()

    all_features = range(M)
    for r in range(M+1):  # r = subset size
        for subset in itertools.combinations(all_features, r):
            s_key = frozenset(subset)
            f_cache[s_key] = f_of_subset(subset)

    shap_values = np.zeros(M, dtype=float)

    # Shapley formula: sum_{S not containing i} [ weight(|S|) * (f(S union i) - f(S)) ]
    for i in range(M):
        for r in range(M):  # subsets of size r
            for subset in itertools.combinations([f for f in all_features if f != i], r):
                s_key = frozenset(subset)
                s_union_i_key = frozenset(subset).union({i})
                subset_size = r  # i.e. |S|
                # contribution = [ f(S union i) - f(S) ] * shapley_weight(|S|)
                contribution = (
                    f_cache[s_union_i_key] 
                    - f_cache[s_key]
                ) * shapley_weight(subset_size)
                shap_values[i] += contribution

    return shap_values
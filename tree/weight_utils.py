import numpy as np
from scipy.special import gamma
import numba
import math

@numba.jit(nopython=True)
def factorial(n):
    """
    Helper: A simple factorial for n >= 0, used for calc_weight.
    """
    if n < 2:
        return 1.0
    result = 1.0
    for i in range(2, n + 1):
        result *= i
    return result

def beta_function(alpha, beta):
    """Helper: Calculate the Beta function."""
    return gamma(alpha) * gamma(beta) / gamma(alpha + beta)

"""
1. Random p generation function
2. Allow user to input p generation function and automatically review whether the sum is 1
3. What condition is needed for using the tree-based algorithm?
"""


def w_shapley(V, U):
    """Calculate the weight of Shapley value."""
    num = factorial(U) * factorial(V - U - 1)
    den = factorial(V)
    return num / den

def w_beta_shapley(V, U, alpha, beta):
    """Calculate the weight of beta Shapley value."""
    num = beta_function(U + beta - 1, V - U + alpha)
    den = beta_function(alpha, beta)
    # print("V, U, alpha, beta, num, den:", V, U, alpha, beta, num, den)
    w = num / den * V
    shap_term = factorial(V - 1) / (factorial(U - 1) * factorial(V - U))
    # print("w, shap_term:", w, shap_term)
    return w * shap_term

def w_banzhaf (V):
    """Calculate the weight of Banzhaf value."""
    return 1 / (2 ** (V - 1))

def w_weighted_banzhaf(V, weight):
    """Calculate the weight of weighted Banzhaf value."""
    return (weight**V) * ((1 - weight)**((V - 1) - V))


# print(w_beta_shapley(10, 6, 1, 1))
# print(w_weighted_banzhaf(10, 0.5))
# print(w_banzhaf(10))



#  --------------------------------------------------------------------------------------------
from itertools import combinations

def weighted_banzhaf(f, n, w): #demo
    phi = [0.0]*n
    
    all_players = set(range(n))
    
    # For each player i, sum over all subsets S of N \ {i}
    for i in range(n):
        other_players = all_players - {i}
        for size in range(len(other_players)+1):
            for S in combinations(other_players, size):
                S = set(S)
                # Marginal contribution f(S U {i}) - f(S)
                marginal_contribution = f(S | {i}) - f(S)
                # Weighted coefficient: w^|S| * (1-w)^(n-1 - |S|)
                weight_factor = (w**size) * ((1 - w)**((n - 1) - size))
                
                phi[i] += marginal_contribution * weight_factor
    
    return phi

# if __name__ == "__main__":
#     def f(S):
#         return len(S)

#     n = 3
#     w = 0.5

#     phi_values = weighted_banzhaf(f, n, w)
#     print("Weighted Banzhaf values:", phi_values)
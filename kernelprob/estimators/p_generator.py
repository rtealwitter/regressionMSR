import numpy as np
from math import factorial
from scipy.special import beta as beta_function
from scipy.special import comb as scipy_comb 


# 1) Randomly generate p that satisfies sum_{k=0}^{n-1} binom(n-1,k)* p[k] = 1
def random_probabilistic_distribution(n, random_state=None):
    """
    Generate a random length-n array p = [p_0, ..., p_{n-1}]
    s.t. sum_{k=0}^{n-1} binom(n-1, k) * p[k] = 1 and each p[k] >= 0.

    Parameters
    ----------
    n : int
        Number of components in the distribution (p_0 through p_{n-1}).
    random_state : int or None
        If not None, sets the np.random.seed for reproducibility.

    Returns
    -------
    p : np.ndarray
        A length-n array of probabilities satisfying the probabilistic value constraints.
    """
    if random_state is not None:
        np.random.seed(random_state)

    x = np.random.rand(n)

    # Compute binomial coefficients for n-1 choose k, k=0..n-1
    binom_coeffs = scipy_comb(n - 1, np.arange(n))

    # Compute the normalizing denominator: sum(binom(n-1, k) * x[k] for k)
    denominator = np.dot(binom_coeffs, x)
    
    if denominator == 0:
        raise ValueError("Denominator in normalization is zero. Choose a different random seed or check n.")

    p = x / denominator

    return p


# 2) Distributions p for four special semivalues
def shapley_distribution(n):
    """
    Generate the distribution p_k for the Shapley value.
    The standard semivalue weight is: w(k,n) = k! * (n - k -1)! / n!
    Then p_k = w(k,n) / binom(n-1, k).

    Parameters
    ----------
    n : int
        Total number of players.

    Returns
    -------
    p : np.ndarray
        Shapley distribution of length n (p_0, p_1, ..., p_{n-1}).
    """
    k = np.arange(n)
    # w(k,n) = k! * (n - k -1)! / n!
    # Handle the case when n - k -1 < 0 by setting w_k to 0
    with np.errstate(over='ignore'):
        p_k = np.array([factorial(ki) * factorial(n - ki - 1) / factorial(n) if (n - ki -1) >=0 else 0.0 for ki in k])

    return p_k

def banzhaf_distribution(n):
    """
    Generate the distribution p_k for the Banzhaf value.
    The standard semivalue weight is: w(k,n) = 1 / 2^(n-1)
    Then p_k = w(k,n) / binom(n-1, k).

    Parameters
    ----------
    n : int
        Total number of players.

    Returns
    -------
    p : np.ndarray
        Banzhaf distribution of length n (p_0, p_1, ..., p_{n-1}).
    """
    return np.full(n, 1.0 / (2 ** (n - 1)))

def beta_shapley_distribution(n, alpha, beta):
    """
    Generate the distribution p_k for the Beta Shapley value.
    Weight: w(k,n) = Beta(k + beta, n - k -1 - alpha) / Beta(alpha, beta)

    Then p_k = w(k,n) / binom(n-1, k).

    Parameters
    ----------
    n : int
        Total number of players.
    alpha : float
    beta : float

    Returns
    -------
    p : np.ndarray
        Beta Shapley distribution p_k of length n.
    """
    k = np.arange(n)
    # w(k,n) = Beta(k + beta, (n -1 -k) + alpha) / Beta(alpha, beta)
    num = np.array([
        beta_function(ki + beta, (n - 1 - ki) + alpha)
        for ki in k
    ])
    den = beta_function(alpha, beta)

    with np.errstate(divide='ignore', invalid='ignore'):
        p_k = np.where(den != 0, num / den, 0.0)

    return p_k

def weighted_banzhaf_distribution(n, weight):
    """
    Generate the distribution p_k for a Weighted Banzhaf value with parameter p_weight.
    Weight: w(k,n) = p_weight^(k-1) * (1 - p_weight)^(n - k)

    Then p_k = w(k,n) / binom(n-1, k).

    Parameters
    ----------
    n : int
        Total number of players.
    p_weight : float
        "p" from the table, 0 <= p_weight <= 1

    Returns
    -------
    p : np.ndarray
        Weighted Banzhaf distribution p_k of length n.
    """
    if not (0 <= weight <= 1):
        raise ValueError("p_weight must be between 0 and 1.")

    k = np.arange(n)

    # w(k,n) = p_weight^(k-1) * (1 - p_weight)^(n - k)
    p_k = weight ** (k - 1) * (1 - weight) ** (n - k)

    return p_k


# Test
if __name__ == "__main__":
    n = 5
    print("Example: random_probabilistic_distribution(n=5):")
    p_rand = random_probabilistic_distribution(n, random_state=42)
    print("p_rand =", p_rand)
    print("Check sum of binom(n-1,k)*p_k =", np.dot(scipy_comb(n-1, np.arange(n)), p_rand))

    print("\nShapley distribution (n=5):")
    p_shap = shapley_distribution(n)
    print("Shapley p =", p_shap)
    print("Check sum of binom(n-1,k)*p_k =", np.dot(scipy_comb(n-1, np.arange(n)), p_shap))

    print("\nBanzhaf distribution (n=5):")
    p_bz = banzhaf_distribution(n)
    print("Banzhaf p =", p_bz)
    print("Check sum of binom(n-1,k)*p_k =", np.dot(scipy_comb(n-1, np.arange(n)), p_bz))

    alpha_val = 1.0
    beta_val = 4.0
    print(f"\nBeta Shapley distribution (n=5, alpha={alpha_val}, beta={beta_val}):")
    p_beta = beta_shapley_distribution(n, alpha_val, beta_val)
    print("Beta Shapley p =", p_beta)
    print("Check sum of binom(n-1,k)*p_k =", np.dot(scipy_comb(n-1, np.arange(n)), p_beta))

    weight = 0.5
    print(f"\nWeighted Banzhaf distribution (n=5, p={weight}):")
    p_wbz = weighted_banzhaf_distribution(n, weight)
    print("Weighted Banzhaf p =", p_wbz)
    print("Check sum of binom(n-1,k)*p_k =", np.dot(scipy_comb(n-1, np.arange(n)), p_wbz))
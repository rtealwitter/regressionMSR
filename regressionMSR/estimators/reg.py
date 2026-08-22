import numpy as np
from scipy.special import comb

import sklearn.ensemble
import sklearn.linear_model
from ..exact.treeprob import tree_prob

from ..base_estimator import BaseEstimator
from ..utils.p_generator import get_p, is_symmetric

# Combined regression class that supports:
# Leverage score sampling
# Paired sampling and non-paired sampling
# Sampling with replacement
# Constrained regression problem and unconstrained regression problem
#
# Not used by any registered estimator (config.py); kept as a library class.
# The sampling-without-replacement path (a Bernoulli-style per-size draw whose
# constant was found by an absolute-tolerance bisection) was removed in August
# 2026 together with playground.py; see the git history before that commit.

def compute_weight_norm(w, p):
    # Faster implementation with vectorized operations
    n = len(p)
    binom_coeffs = comb(n - 1, np.arange(n))  # Vector of binomial coefficients
    w_inv_sq = 1 / w[:n]**2 + 1 / w[1:n+1]**2  # Vectorized inverse squares
    c_sq = np.sum(binom_coeffs * p**2 * w_inv_sq)
    return np.sqrt(c_sq)

def compute_weights(w1, p):
    n = len(p)
    w = np.zeros(n+1)
    w[0] = 1
    w[1] = w1
    # Solve for the rest of the weights
    for ell in range(1, n):
        numerator = p[ell]**2 * w[ell]**2 * w[ell-1]**2
        denominator1 = 2 * p[ell] * p[ell-1] * w[ell-1]**2
        denominator2 = -1*p[ell-1]**2 * w[ell]**2
        w_sq = numerator / (denominator1 + denominator2)
        if w_sq < 0: return np.full(n+1, np.nan)
        w[ell+1] = np.sqrt(w_sq)

    c = compute_weight_norm(w, p)
    # Check if c is nan
    #if not np.isnan(c):
    #    assert np.allclose(compute_weight_norm(w*c, p), 1)
    return w*c

def find_w1(p, start=10e-10, stop=10e10, num=1000):
    valid_w1s = []
    objectives = []
    for w1 in np.geomspace(start, stop, num):
        w = compute_weights(w1, p)
        if np.isnan(w).any(): continue
        objectives.append(
            ((w[::-1] - w)**2).sum()
        )
        valid_w1s.append(w1)
    closest_idx = np.argmin(np.array(objectives))
    return valid_w1s[closest_idx]

class UniversalRegression(BaseEstimator):
    def __init__(
        self,
        model,
        baseline,
        weighting,
        constrain_reg=True,
        seed=None,
    ):
        """
        Additional Parameters
        ----------
        constrain_reg:
            Whether to solve the constrained regression problem.
        seed:
            Seed for the sampling generator (None: fresh entropy, not replayable).
        Subsets are sampled with replacement, in pairs (S and its complement)
        whenever the probabilistic weights are symmetric.
        """
        super().__init__(model, baseline, weighting)
        self.n = self.baseline.shape[1] # Number of features
        self.p = get_p(self.n, weighting)
        self.model = model
        self.baseline = baseline
        self.gen = np.random.Generator(np.random.PCG64(seed))

        # Check if self.p is symmetric
        self.pair_sampling = is_symmetric(self.p)

        self.constrain_reg = constrain_reg
    
    def explain(self, explicand, num_samples, true_sum=None):
        self.explicand = explicand
        self.true_sum = true_sum

        if self.constrain_reg:
            self.num_samples = (num_samples -2 ) // 2 * 2 # Two used for v1,v0
            self.valid_sizes = np.array(list(range(1, self.n)))
            self.leverage_score = lambda s : (self.p[s] + self.p[s-1]) * s * (self.n - s)
#            self.leverage_score = lambda s : np.ones_like(s)

            self.sample_with_replacement()
            self.compute_constrained()
        else:            
            self.num_samples = num_samples//2 * 2
            self.load_unconstrained()
            ## Add dimension to p for faster operations
            #self.p = np.append(self.p, -1e10)

            self.sample_with_replacement()
            self.compute_unconstrained()

        return self.phi
    
    def load_unconstrained(self):
        self.valid_sizes = np.array(list(range(self.n+1)))

        w1 = find_w1(self.p)
        self.w = compute_weights(w1, self.p)
#        print(f"w1: {w1}")
#        print(f"w: {self.w}")

        def leverage_score(s):
            p_ext = np.append(self.p, 0)
            lev = 0
            lev += (s > 0) * s * p_ext[s-1]**2
            lev += (s < self.n) * (self.n - s) * p_ext[s]**2
            lev /= self.w[s] ** 2
            return lev
        
#        leverage_score = lambda s : np.ones_like(s)

#        sum1, sum2 = 0, 0
#        for ell in range(self.n):
#            sum1 += scipy.special.binom(self.n-1, ell) * self.p[ell]**2
#            if ell >= 1:
#                sum2 += scipy.special.binom(self.n-2, ell-1) * (self.p[ell]-self.p[ell-1])**2
#        self.an = 2*sum1 - sum2
#        self.bn = sum2 / self.an
#    
#        def leverage_score(s):
#            first_term = 0 # |S| p_{|S|-1}^2 + (n-|S|) p_{|S|}^2
#            second_term = 0 # |S| p_{|S|-1} - (n-|S|) p_{|S|}
#            first_term += (s > 0) * s * self.p[s-1]**2
#            second_term += (s > 0) * s * self.p[s-1]
#            first_term += (s < self.n) * (self.n - s) * self.p[s]**2
#            second_term -= (s < self.n) * (self.n - s) * self.p[s]
#            #if s > 0:
#            #    first_term += s * self.p[s-1]**2
#            #    second_term += s * self.p[s-1]
#            #if s < self.n:
#            #    first_term += (self.n - s) * self.p[s]**2
#            #    second_term -= (self.n - s) * self.p[s]
#            # (first_term - bn / (1+n * bn) * second_term^2) / a_n
#            return (first_term - self.bn / (1+self.n * self.bn) * second_term**2) / self.an

        self.leverage_score = lambda s : np.array([leverage_score(val) for val in s])

    def update_prob(self):
        self.prob = self.leverage_score(self.valid_sizes)
        self.prob /= np.sum(self.prob)
        
    def get_probs(self, s):
        if self.constrain_reg: # Length 1 is first prob
            return self.prob[s-1]
        return self.prob[s]
    
    def add_one_sample(self, idx, indices):
        if not self.pair_sampling:
            self.sampled[idx, indices] = 1
        else:            
            self.sampled[2*idx, indices] = 1
            indices_complement = np.setdiff1d(np.arange(self.n), indices)  # int dtype even when empty (size n)
            self.sampled[2*idx+1, indices_complement] = 1
    
    def sample_with_replacement(self):
        self.sampled = np.zeros((self.num_samples, self.n))
        num_sizes = self.num_samples if not self.pair_sampling else self.num_samples // 2
        self.update_prob() # Important to update self.prob
        sampled_sizes = self.gen.choice(self.valid_sizes, num_sizes, p=self.prob)
        for idx, s in enumerate(sampled_sizes):
            indices = self.gen.choice(self.n, s, replace=False)
            self.add_one_sample(idx, indices)
             
    def compute_unconstrained(self):
        # x* = (A^T A)^-1 A^T v
        # xt = (A^T S^T W S A)^-1 A^T S^T W S v
        # phit = (1+bn 1 1^T) xt
        sizes = np.sum(self.sampled, axis=1).astype(int)
        self.sizes = sizes
        p_ext = np.append(self.p, 0)
        
        # Build A_tilde so A_tilde[S,i] = p[|S|-1]/w[|S|] if i in S else -p[|S|]/w[|S|]
        A_tilde = p_ext[sizes-1][:,None] * self.sampled - p_ext[sizes][:,None] * (1-self.sampled)
        A_tilde /= self.w[sizes][:,None]

        inputs = self.baseline * (1 - self.sampled) + self.explicand * self.sampled
        self.y = self.model.predict(inputs)
        Sv = self.y * self.w[sizes]
        self.prob_sampled = self.get_probs(sizes)

        reg_weighting = 1/self.prob_sampled

        ASSv = A_tilde.T @ np.diag(reg_weighting) @ Sv
        ASSA = A_tilde.T @ np.diag(reg_weighting) @ A_tilde
        self.phi = np.linalg.lstsq(ASSA, ASSv, rcond=None)[0]

    def compute_constrained(self):
        # A = Z P
        # b = v(z) - v0 - Z1 * sum(phi) / n
        # (A^T S^T S A)^-1 A^T S^T S b + (v1 - v0) / n
        # (P^T Z^T S^T S Z P)^-1 P^T Z^T S^T S b + (v1 - v0) / n

        self.sampled = self.sampled[np.sum(self.sampled, axis=1) != 0] # Remove zero rows
        sampled = self.sampled
        v0, v1 = self.model.predict(self.baseline), self.model.predict(self.explicand)
        self.v0, self.v1 = v0, v1
        inputs = self.baseline * (1 - sampled) + self.explicand * sampled
        self.y = self.model.predict(inputs)
        Sv = self.y - v0

        # Estimate constant term = 1/n sum_i phi_i
        # = 1/n sum_S (v(S)-v(emptyset)) (-p[|S|]*(n-|S|) + p[|S|-1] * |S|)
        # Compute probability each set size was sampled
        self.sizes = np.sum(sampled, axis=1).astype(int)
        sizes = self.sizes
        self.prob_sampled = self.get_probs(sizes)
        prob_sampled = self.prob_sampled


        sum_weighting = -self.p[sizes] * (self.n - sizes) + self.p[sizes-1] * sizes
        sum_phi = prob_sampled * sum_weighting @ Sv
        sum_phi = sum_phi + (v1-v0)*self.p[-1]*self.n

        if self.true_sum is not None:
            sum_phi = self.true_sum
        
        Sb = Sv - sizes * sum_phi / self.n

        # Projection matrix
        P = np.eye(self.n) - 1/self.n * np.ones((self.n, self.n))

        reg_weighting = 1/prob_sampled * (self.p[sizes-1] + self.p[sizes])

        PZSSb = P @ sampled.T @ np.diag(reg_weighting) @ Sb
        PZSSZP = P @ sampled.T @ np.diag(reg_weighting) @ sampled @ P
        PZSSZP_inv_PZSSb = np.linalg.lstsq(PZSSZP, PZSSb, rcond=None)[0]

        self.phi = PZSSZP_inv_PZSSb 
        self.phi += sum_phi / self.n

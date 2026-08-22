import numpy as np
import scipy
from scipy.special import comb
import scipy.special
import pandas as pd

import sklearn.ensemble
import sklearn.linear_model
from ..exact.treeshap import tree_shap
from ..exact.treeprob import tree_prob
from ..exact.enumeration import enumeration_prob
from ..estimators.reg import UniversalRegression

from ..base_estimator import BaseEstimator
from ..utils.p_generator import get_p, is_symmetric
from .est_utils import combination_generator
import xgboost

class NullModel:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.zeros(X.shape[0])

class LinearModel:
    def __init__(self, phi):
        self.coef_ = phi
    
    def fit(self, X, y):
        pass

    def predict(self, X):
        return X.values @ self.coef_

def leverage_shap(X, y, cache):
    prob_sampled, v1, v0, p = cache['prob_sampled'], cache['v1'], cache['v0'], cache['p']
    n = X.shape[1]
    sizes = X.sum(axis=1).astype(int)
    Sv = y - v0
    # This function is only reached for Shapley weights (get_fit routes 'shapley' and
    # 'beta_shapley_1_1' here), for which sum_weighting vanishes identically and p[-1] * n = 1,
    # so sum_phi below is exactly the efficiency constraint v1 - v0.  (For general p the first
    # term would have to be a Horvitz-Thompson sum, i.e. divide by prob_sampled, not multiply.)
    sum_weighting = - p[sizes] * (n - sizes) + p[sizes-1] * sizes
    sum_phi = prob_sampled * sum_weighting @ Sv
    sum_phi = sum_phi + (v1 - v0) * p[-1] * n

    Sb = Sv - sizes * sum_phi / n
    P = np.eye(n) - 1/n
    reg_weighting = 1/prob_sampled * (p[sizes-1] + p[sizes])
    PZSSb = P @ X.T @ np.diag(reg_weighting) @ Sb
    PZSSZP = P @ X.T @ np.diag(reg_weighting) @ X @ P
    PZZSZP_inv_PZSSb = np.linalg.lstsq(PZSSZP, PZSSb, rcond=None)[0]

    reg_phi = PZZSZP_inv_PZSSb + sum_phi / n
    return LinearModel(reg_phi), reg_phi    

def kernel_banzhaf(X, y):
    reg_phi = np.linalg.lstsq(X - 1/2, y, rcond=None)[0]
    return LinearModel(reg_phi), reg_phi

def get_fit(X_flat, y_flat, weighting, regression_adj, cache=None):
    n = X_flat.shape[1]
    if regression_adj is False:
        reg_model = NullModel()
        phi_method = lambda reg_model : np.zeros(n)
    elif regression_adj == "linear" and weighting in ['shapley', 'beta_shapley_1_1']:
        return leverage_shap(X_flat.values, y_flat, cache)
    elif regression_adj == "linear" and weighting in ['banzhaf', 'weighted_banzhaf_0.5']:
        return kernel_banzhaf(X_flat.values, y_flat)
    elif regression_adj == "linear":
        reg_model = sklearn.linear_model.LinearRegression()
        phi_method = lambda reg_model: reg_model.coef_
    elif regression_adj == "tree":
        reg_model = xgboost.XGBRegressor()
        if weighting in ['shapley', 'beta_shapley_1_1']:
            phi_method = lambda reg_model: tree_shap(np.zeros((1,n)), np.ones((1,n)), reg_model)[0].squeeze()
        else:
            phi_method = lambda reg_model: tree_prob(np.zeros((1,n)), np.ones((1,n)), reg_model, weighting)[0].squeeze()
    else:
        raise ValueError("regression_adjustment must be False, 'linear', or 'tree'")

    reg_model.fit(X_flat, y_flat)
    phi = phi_method(reg_model)

    return reg_model, phi

class UniversalMSR(BaseEstimator):
    def __init__(self, model, baseline, weighting, reg_model_class=False, return_direct=False):
        super().__init__(model, baseline, weighting)
        self.n = self.baseline.shape[1]
        self.p = get_p(self.n, weighting)
        self.model = model
        self.baseline = baseline
        self.gen = np.random.Generator(np.random.PCG64())
        self.reg_model_class = reg_model_class
        self.valid_sizes = np.arange(self.n+1)
        self.pair_sampling = is_symmetric(self.p)
        self.v1 = self.v0 = None        
        self.adjust = 1 if (self.reg_model_class == 'linear' and self.weighting in ['shapley', 'beta_shapley_1_1']) else 0

        self.sample_dist = []
        for size in range(self.n+1):
            prob = 0
            if size > 0:
                prob += self.p[size-1]**2 * size
            if size < self.n:
                prob += self.p[size]**2 * (self.n - size)
            self.sample_dist.append( np.sqrt(prob) )
        

        if reg_model_class == 'linear':
            # Leverage SHAP
            if weighting in ['shapley', 'beta_shapley_1_1']:
                s = np.arange(1, self.n)
                self.sample_dist = (self.p[s] + self.p[s-1]) * s * (self.n- s)
                self.valid_sizes = s
            # Kernel Banzhaf
            if weighting in ['banzhaf', 'weighted_banzhaf_0.5']:
                self.sample_dist = np.ones_like(self.sample_dist)

        self.sample_dist = np.array(self.sample_dist) / np.sum(self.sample_dist)
        self.return_direct = return_direct
    
    def add_sample(self, idx, indices, prob):
        if not self.pair_sampling:
            self.X[idx, indices] = 1
            self.actual_sample_prob[idx] = prob
        else:
            self.X[2*idx, indices] = 1
            self.actual_sample_prob[2*idx] = prob
            complement_indices = np.setdiff1d(np.arange(self.n), indices)
            self.X[2*idx + 1, complement_indices] = 1
            self.actual_sample_prob[2*idx + 1] = prob
       
    def sample_with_replacement(self, num_samples):
        self.X = np.zeros((num_samples, self.n), dtype=float)
        self.actual_sample_prob = np.zeros(num_samples, dtype=float)
        elements = np.arange(self.n)
        idx = 0
        sample_dist_size = [prob * scipy.special.binom(self.n, size) for size, prob in zip(self.valid_sizes, self.sample_dist)]
        sample_dist_size = np.array(sample_dist_size) / np.sum(sample_dist_size)
        num_pairs = num_samples // 2 if self.pair_sampling else num_samples
        sampled_sizes = self.gen.choice(self.valid_sizes, num_pairs, p=sample_dist_size)

        # actual_sample_prob stores the EXPECTED MULTIPLICITY of each sampled set S
        # over the whole sample, i.e. m * D(S), so that sum_{sampled S} g(S) / actual_sample_prob
        # is an unbiased estimate of sum_S g(S) (same semantics as the inclusion probability
        # stored by sample_without_replacement).  D(S) = sample_dist[|S|] / Z with
        # Z = sum_s sample_dist[s] * binom(n, s); the sampled-size probabilities above are
        # exactly sample_dist[s] * binom(n, s) / Z.  (Previously the unnormalised
        # sample_dist[|S|] was stored, which is off from D(S) by the constant Z.)
        Z = np.sum([prob * scipy.special.binom(self.n, size) for size, prob in zip(self.valid_sizes, self.sample_dist)])
        for idx, size in enumerate(sampled_sizes):
            indices = self.gen.choice(elements, size=size, replace=False)
            self.add_sample(idx, indices, num_samples * self.sample_dist[size - self.adjust] / Z)
        
    def _strata(self, C):
        """Per-size targets of the stratified without-replacement design for the constant C.

        Returns a list of (size, middle, population, x_units): for every valid size s (only
        s <= n/2 under pair_sampling) the number of UNITS available and the TARGET number of
        units, where a unit is one size-s subset or, under pair_sampling, a pair
        {S, complement(S)} with |S| = s.  The target is
            x_s = min(C * sample_dist[s], 1) * binom(n, s)
        subsets of size s; for the paired middle stratum of an even n (units: the pairs whose S
        contains element n-1, binom(n-1, s-1) of them) that is x_s / 2 units.
        """
        sample_dist_C = np.minimum(self.sample_dist * C, np.ones_like(self.sample_dist))
        strata = []
        for s, prob in zip(self.valid_sizes, sample_dist_C):
            x_s = prob * scipy.special.binom(self.n, s)          # target number of size-s subsets
            middle = self.pair_sampling and s == self.n // 2 and self.n % 2 == 0
            if middle:
                strata.append((s, True, scipy.special.binom(self.n - 1, s - 1), x_s / 2))
            else:
                strata.append((s, False, scipy.special.binom(self.n, s), x_s))
            if self.pair_sampling and s == self.n // 2:
                break
        return strata

    @staticmethod
    def _units_drawn(population, x_units):
        """Units actually drawn for a target of x_units: round(x_units), but at least one (every
        size is represented, so every subset has a positive inclusion probability) and at most
        the population (the size is then enumerated)."""
        return int(min(population, max(1, round(x_units))))

    def find_bernoulli_constant(self, num_samples):
        """Constant C for which the design of _strata(C) has about num_samples rows."""
        rows_per_unit = 2 if self.pair_sampling else 1
        m = min(num_samples, 2**self.n - 2 * self.adjust)

        def rows(C):   # continuous relaxation of the rounded design, nondecreasing in C
            return rows_per_unit * sum(min(population, max(1.0, x_units))
                                       for _, _, population, x_units in self._strata(C))

        C = 1 / 2**self.n
        L = 1 / 100**self.n
        R = 100**self.n
        expected = rows(C)
        while round(expected) != m and R - L > 1e-6:
            if expected < m: L = C
            else: R = C
            C = (L + R) / 2
            expected = rows(C)
        return C

    def sample_without_replacement(self, num_samples):
        """Stratified-by-size sample of DISTINCT subsets, about num_samples rows in total.

        For every valid size s the target number of size-s subsets is
            x_s = min(C * sample_dist[s], 1) * binom(n, s),
        with C from find_bernoulli_constant, and m_s = max(1, round(x_s)) of them -- all of them
        once m_s reaches binom(n, s) -- are drawn uniformly without replacement, independently
        across sizes.  Under pair_sampling a unit is a pair {S, complement(S)}, only sizes
        s <= n/2 are drawn, and the complement rows cover the sizes above n/2.
        actual_sample_prob stores the REALISED inclusion probability of every row's unit,
        m_s / binom(n, s) (m_s / binom(n-1, s-1) for the paired middle stratum of an even n), so
            sum_{sampled S} g(S) / actual_sample_prob[S]
        is an unbiased estimate of sum_S g(S) for every g.

        Returns, per stratum, (population, num_units, rows): the number of units available, the
        number drawn, and the list of row index arrays of the drawn units (one array per unit).
        """
        C = self.find_bernoulli_constant(num_samples)
        strata = [(s, middle, population, self._units_drawn(population, x_units))
                  for s, middle, population, x_units in self._strata(C)]
        rows_per_unit = 2 if self.pair_sampling else 1
        num_rows = rows_per_unit * sum(num_units for _, _, _, num_units in strata)
        self.X = np.zeros((num_rows, self.n), dtype=float)
        self.actual_sample_prob = np.zeros(num_rows, dtype=float)

        units = []
        idx = 0
        for s, middle, population, num_units in strata:
            pi_s = num_units / population                       # realised inclusion probability
            if middle:
                draws = (list(indices) + [self.n - 1]
                         for indices in combination_generator(self.gen, self.n - 1, s - 1, num_units))
            else:
                draws = combination_generator(self.gen, self.n, s, num_units)
            rows = []
            for indices in draws:
                self.add_sample(idx, list(indices), pi_s)
                rows.append(np.arange(rows_per_unit * idx, rows_per_unit * (idx + 1)))
                idx += 1
            units.append((population, num_units, rows))
        return units

    def sample_folds_without_replacement(self, num_samples, k):
        """One stratified sample without replacement, with a held-out part for each of k folds.

        The sample is that of sample_without_replacement(num_samples).  Within every size the
        drawn units are split into k parts as evenly as possible, in random order.  If fewer than
        k units of a size were drawn, every fold whose part is empty holds out one drawn unit of
        that size chosen uniformly at random instead -- unless the size is fully enumerated, in
        which case there is nothing left to extrapolate to and such a fold holds out nothing of
        that size.  Fold l's training set is the sample minus its held-out part.

        Returns (folds, fold_prob): folds[l] holds the row indices of fold l's held-out part and
        fold_prob[l] the matching CONDITIONAL inclusion probabilities
            t / (N - m + t),
        where, for the row's size, N is the number of units available, m the number drawn and t
        the number fold l holds out.  Given fold l's training set (and hence anything fitted to
        it), the held-out units of a size are a uniform sample without replacement of size t
        from the N - m + t units the training set does not contain -- every (training part,
        held-out part) of the right sizes is equally likely -- so
            sum_{training rows} g(S)  +  sum_{held-out rows} g(S) / fold_prob
        is, conditionally on the training set, an unbiased estimate of sum_S g(S) for every g:
        the training subsets are counted exactly, and the held-out subsets stand in for the
        subsets nobody drew.  For a g that is the same for every fold, the average of this over
        the k folds is exactly the single-sample estimate
        sum_{sampled S} g(S) / actual_sample_prob[S] whenever every size is enumerated or its
        number of drawn units is an exact multiple of k (equal held-out parts in every fold), so
        cross-fitting costs nothing in that case.  (At least k drawn units is not enough: with
        unequal parts the fold weights 1/fold_prob differ between folds and the average is only
        unbiased, not identical, e.g. n=9, k=3, m=27.)
        """
        units = self.sample_without_replacement(num_samples)
        folds = [[] for _ in range(k)]
        fold_prob = [[] for _ in range(k)]
        for population, num_units, rows in units:
            parts = np.array_split(self.gen.permutation(num_units), k)   # sizes differ by <= 1
            enumerated = num_units >= population
            for part, l in zip(parts, self.gen.permutation(k)):          # random fold order
                if len(part) == 0:
                    if enumerated:
                        continue
                    part = self.gen.integers(num_units, size=1)
                t = len(part)
                pi = t / (population - num_units + t)
                for u in part:
                    folds[l].extend(rows[u])
                    fold_prob[l].extend([pi] * len(rows[u]))
        folds = [np.array(f, dtype=int) for f in folds]
        fold_prob = [np.array(p, dtype=float) for p in fold_prob]
        return folds, fold_prob

    def explain(self, explicand, num_samples, k=10, with_replacement=None):
        """Estimate the attribution of explicand from about num_samples evaluations, k-fold cross-fitted.

        with_replacement=None (default) draws distinct subsets for n < 50 and i.i.d. subsets
        otherwise; pass True/False to force a sampling path (used by the audit scripts).
        """
        if with_replacement is None:
            with_replacement = self.n >= 50
        if num_samples >= 2**self.n:
            return enumeration_prob(self.baseline, explicand, self.model, self.weighting)
        if self.pair_sampling:
            num_samples = num_samples // (2*k) * (2*k)
        else:
            num_samples = num_samples // k * k
        if num_samples == 0:
            raise ValueError(
                f'num_samples too small for k={k} folds'
                f'{" with paired sampling" if self.pair_sampling else ""}: need at least {2*k if self.pair_sampling else k}'
            )

        if self.reg_model_class == 'linear' and self.weighting in ['shapley', 'beta_shapley_1_1']:
            # scalars: np.asarray(...).reshape(-1)[0] works for (1,), (1,1) and 0-d outputs
            self.v1 = float(np.asarray(self.model.predict(explicand)).reshape(-1)[0])
            self.v0 = float(np.asarray(self.model.predict(self.baseline)).reshape(-1)[0])

        # Cross-fitting.  Fold l's surrogate f_l is fitted to the rows outside fold l's held-out
        # part, and the residual correction for fold l must be unbiased CONDITIONALLY on f_l.
        #   Without replacement (n < 50): one stratified sample, split into held-out parts.  The
        #     correction counts the training rows' residuals exactly and extrapolates from the
        #     held-out rows only to the subsets nobody drew, with the conditional inclusion
        #     probabilities of sample_folds_without_replacement.  (Extrapolating from the held-out
        #     rows to ALL subsets with the unconditional probabilities is biased for a surrogate
        #     that fits its training points closely -- a tree -- because a held-out subset is by
        #     construction one f_l never saw; drawing the k folds as independent samples is
        #     unbiased but forfeits the enumeration of the heavy small and large sizes that a
        #     single sample of the full budget affords.)
        #   With replacement (n >= 50): i.i.d. draws, randomly partitioned; each part is independent of the others,
        #     so the plain Horvitz-Thompson sum over the held-out part is unbiased given f_l.
        # fold_prob[l][j] is the (conditional) inclusion probability / expected multiplicity of
        # held-out row folds[l][j] in fold l; fold_weights sum to 1.
        if not with_replacement:
            folds, fold_prob = self.sample_folds_without_replacement(num_samples, k)
            fold_weights = [1 / k] * k
            count_training_rows = True
        else:
            self.sample_with_replacement(num_samples)
            num_rows = self.X.shape[0]
            # Randomly partition ALL rows into k folds (sizes differing by at most one pair)
            # while keeping pairs together; every sampled set lands in exactly one fold.
            if self.pair_sampling:
                pair_ids = self.gen.permutation(num_rows // 2)
                folds = [np.reshape(np.column_stack((2 * ids, 2 * ids + 1)), -1)
                         for ids in np.array_split(pair_ids, k)]
            else:
                folds = np.array_split(self.gen.permutation(num_rows), k)
            fold_weights = [len(indices) / num_rows for indices in folds]
            # actual_sample_prob is num_rows * D(S); within a fold of size |fold| the expected
            # multiplicity of S is |fold| * D(S).
            fold_prob = [self.actual_sample_prob[indices] * w_fold
                         for indices, w_fold in zip(folds, fold_weights)]
            count_training_rows = False
        num_rows = self.X.shape[0]
        model_input = self.baseline * (1 - self.X) + explicand * self.X
        y = self.model.predict(model_input)

        # coef[S, i] = p_{|S|-1} 1[i in S] - p_{|S|} 1[i not in S], the weight of v(S) in phi_i(v)
        # (p_{-1} and p_n are never used: they multiply an indicator that is zero).
        sizes = self.X.sum(axis=1).astype(int)
        p_below = np.concatenate(([0.0], self.p))[sizes]
        p_above = np.concatenate((self.p, [0.0]))[sizes]
        coef = self.X * p_below[:, None] - (1 - self.X) * p_above[:, None]

        phi = np.zeros(self.n)
        feature_names = [f'feature_{j}' for j in range(self.n)]
        for indices, pi, w_fold in zip(folds, fold_prob, fold_weights):
            train = np.ones(num_rows, dtype=bool)
            train[indices] = False
            X_train = pd.DataFrame(self.X[train], columns=feature_names)
            X_test = pd.DataFrame(self.X[indices], columns=feature_names)

            # For leverage SHAP
            cache = {
                'prob_sampled': self.actual_sample_prob[train],
                'v1': self.v1,
                'v0': self.v0,
                'p': self.p
            }
            reg_model, reg_phi = get_fit(
                X_train, y[train], self.weighting, self.reg_model_class, cache
            )

            # Fold estimate:  phi^(l) = phi(f_l) + correction,  with
            #   correction_i = [sum_{training S} r_l(S) coef[S, i]]   (n < 50 only)
            #                + sum_{held-out S} r_l(S) coef[S, i] / fold_prob[S],
            #   r_l(S) = v(S) - f_l(S).
            # Conditionally on f_l this is an unbiased estimate of phi(v - f_l) for ANY surrogate
            # (see sample_folds_without_replacement), so E[phi^(l) | f_l] = phi(v); the final
            # estimate is the weighted average over the folds.
            correction = np.zeros(self.n)
            if len(indices) > 0:
                residual = y[indices] - reg_model.predict(X_test)
                correction += (residual / pi) @ coef[indices]
            if count_training_rows:
                residual_train = y[train] - reg_model.predict(X_train)
                correction += residual_train @ coef[train]
            phi += w_fold * (reg_phi + correction)
            if self.reg_model_class == 'linear' and self.weighting in ['shapley', 'beta_shapley_1_1']:
                # Efficiency correction for the linear surrogate, weighted like the terms above.
                phi += w_fold * (self.v1 - self.v0 - reg_phi.sum()) / self.n

        return phi

class LinearMSR(UniversalMSR):
    def __init__(self, model, baseline, weighting):
        super().__init__(model, baseline, weighting, reg_model_class='linear')

class TreeMSR(UniversalMSR):
    def __init__(self, model, baseline, weighting):
        super().__init__(model, baseline, weighting, reg_model_class='tree')

class Tree(UniversalMSR):
    def __init__(self, model, baseline, weighting):
        super().__init__(model, baseline, weighting, reg_model_class='tree', return_direct=True)

class MSR(UniversalMSR):
    def __init__(self, model, baseline, weighting):
        super().__init__(model, baseline, weighting, reg_model_class=False)

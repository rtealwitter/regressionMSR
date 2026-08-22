"""
AllSamplesMSR: the "Practical Simplification" from the paper (main.tex, the
"Practical Simplification" paragraph) -- train ONE surrogate on ALL sampled
coalitions (instead of k-fold cross-fitting as in UniversalMSR.explain) and
build the MSR estimate with it.

This module does not modify regMSR.py. It reuses the exact same sampler
(UniversalMSR.sample_without_replacement / sample_with_replacement, same
sample_dist, same self.p, same prob_sampled = expected multiplicity of S)
and the exact same Horvitz-Thompson correction formula used inside
UniversalMSR.explain's k-loop (sum, not mean, over the samples containing /
excluding feature i, each weighted by p[.]/prob_sampled) -- just with k=1
and the entire sample acting as both the "training" set and the "held-out"
set (since there is no held-out fold left once all m samples are used to
fit the surrogate). That is precisely the point being stress-tested: if the
surrogate f_hat interpolates its own training coalitions, y - f_hat(S) = 0
for every sampled S and the residual-correction term collapses to zero, so
the estimator degenerates to phi(f_hat) with no correction at all.

Diagnostics exposed after calling explain():
    self.diagnostics = {
        'train_residual_rms': ||y - f_hat(S)|| / sqrt(m)   (in-sample residual RMS)
        'y_rms': ||y|| / sqrt(m)
        'train_residual_ratio': train_residual_rms / y_rms  (~0 => interpolation)
        'correction_norm': || phi - phi_surrogate ||_2       (norm of the MSR
            correction vector that was added on top of the surrogate's own
            attribution phi_surrogate; ~0 => correction vanished)
        'phi_surrogate': phi(f_hat) alone, i.e. the "TreeSurrogateOnly" /
            "LinearSurrogateOnly" estimate with NO residual correction.
    }
"""
import numpy as np
import pandas as pd

from ..exact.enumeration import enumeration_prob
from .regMSR import UniversalMSR, get_fit


class AllSamplesMSR(UniversalMSR):
    """
    Same constructor/sampler as UniversalMSR. explain() fits a single
    surrogate on all m sampled coalitions (k=1, whole sample used as both
    'train' and 'test' fold) instead of k-fold cross-fitting.
    """

    def explain(self, explicand, num_samples):
        if num_samples >= 2 ** self.n:
            phi = enumeration_prob(self.baseline, explicand, self.model, self.weighting)
            phi = np.asarray(phi[0] if isinstance(phi, list) else phi, dtype=float).ravel()
            self.diagnostics = {
                'train_residual_rms': np.nan, 'y_rms': np.nan, 'train_residual_ratio': np.nan,
                'correction_norm': np.nan, 'phi_surrogate': phi.copy(), 'num_samples': 2 ** self.n,
            }
            return phi

        # Mirror UniversalMSR.explain's rounding of num_samples with k=1:
        # pair_sampling needs an even sample count; otherwise no rounding needed.
        if self.pair_sampling:
            num_samples = num_samples // 2 * 2
        if num_samples == 0:
            raise ValueError(
                f'num_samples too small{" with paired sampling" if self.pair_sampling else ""}: need at least '
                f'{2 if self.pair_sampling else 1}'
            )

        if self.reg_model_class == 'linear' and self.weighting in ['shapley', 'beta_shapley_1_1']:
            self.v1 = float(np.asarray(self.model.predict(explicand)).reshape(-1)[0])
            self.v0 = float(np.asarray(self.model.predict(self.baseline)).reshape(-1)[0])

        if self.n < 50:
            self.sample_without_replacement(num_samples)
        else:
            self.sample_with_replacement(num_samples)
        num_samples = self.X.shape[0]

        sizes = self.X.sum(axis=1).astype(int)
        model_input = self.baseline * (1 - self.X) + explicand * self.X
        y = self.model.predict(model_input)

        feature_names = [f'feature_{j}' for j in range(self.n)]
        X_train = pd.DataFrame(self.X, columns=feature_names)
        X_test = X_train  # same samples: no held-out fold left (k=1)
        y_train = y

        cache = {
            'prob_sampled': self.actual_sample_prob,
            'v1': self.v1,
            'v0': self.v0,
            'p': self.p,
        }
        reg_model, reg_phi = get_fit(
            X_train, y_train, self.weighting, self.reg_model_class, cache
        )
        reg_pred = reg_model.predict(X_test)

        # ---- Diagnostics: does the surrogate interpolate its training data? ----
        residual = y - reg_pred
        m = num_samples
        train_residual_rms = float(np.linalg.norm(residual) / np.sqrt(m))
        y_rms = float(np.linalg.norm(y) / np.sqrt(m))
        train_residual_ratio = train_residual_rms / y_rms if y_rms > 0 else np.nan

        group = {
            'sampled': self.X,
            'y': y,
            'prob_sampled': self.actual_sample_prob,
            'sizes': sizes,
            'reg_pred': reg_pred,
        }

        phi_surrogate = np.array(reg_phi, dtype=float).copy()
        correction = np.zeros(self.n)

        # Horvitz-Thompson form (matches UniversalMSR.explain exactly, k=1: the "fold"
        # is the whole sample, so phi[idx] = phi_est[idx]/1 + sum_with - sum_without,
        # a SUM (not mean) over the samples containing/excluding feature idx).
        for idx in range(self.n):
            idx_contained = (group['sampled'][:, idx] == 1)
            not_contained = ~idx_contained
            sum_with = np.sum(
                residual[idx_contained] * self.p[sizes[idx_contained] - 1]
                / group['prob_sampled'][idx_contained]
            )
            sum_without = np.sum(
                residual[not_contained] * self.p[sizes[not_contained]]
                / group['prob_sampled'][not_contained]
            )
            correction[idx] = sum_with - sum_without

        if self.reg_model_class == 'linear' and self.weighting in ['shapley', 'beta_shapley_1_1']:
            # Same efficiency correction the parent adds per-fold (divided by k there);
            # added once here since there is only one "fold" (k=1).
            correction += (self.v1 - self.v0 - phi_surrogate.sum()) / self.n

        phi = phi_surrogate + correction

        self.diagnostics = {
            'train_residual_rms': train_residual_rms,
            'y_rms': y_rms,
            'train_residual_ratio': train_residual_ratio,
            'correction_norm': float(np.linalg.norm(correction)),
            'phi_surrogate': phi_surrogate,
            'num_samples': int(m),
            'reg_model': reg_model,  # kept for introspection (e.g. realized XGBoost tree depth); not JSON/CSV-safe
        }

        return phi


class LinearMSRAll(AllSamplesMSR):
    def __init__(self, model, baseline, weighting):
        super().__init__(model, baseline, weighting, reg_model_class='linear')


class TreeMSRAll(AllSamplesMSR):
    def __init__(self, model, baseline, weighting):
        super().__init__(model, baseline, weighting, reg_model_class='tree')

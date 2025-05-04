# wshap_perm.py   – Weighted‑SHAP (permutation) estimator
import numpy as np
import math

from ..base_estimator import BaseEstimator
from ..utils.p_generator import get_p


class WeightedSHAPEstimator(BaseEstimator):
    """
    Permutation‑based Weighted‑SHAP estimator.

        φ_i  =  ∑_{k=0}^{n-1}  m_k · E[ U(S_k ∪ {i}) − U(S_k) ]
        m_k  =  C(n-1, k) · p_k

    """

    def __init__(self, model, baseline: np.ndarray, weighting: str):
        super().__init__(model, baseline, weighting)

        self.n_feat = self.baseline.size if self.baseline.ndim == 1 else self.baseline.shape[1]
        p = get_p(self.n_feat, weighting)                       # length n
        combs = np.array([math.comb(self.n_feat - 1, k) for k in range(self.n_feat)],
                 dtype=float)
        self.m_vec = self.n_feat * p * combs                            #  m_k = C(n-1,k) p_k
        self.base_vec = self.baseline.reshape(-1).astype(float)
        self.rng = np.random.default_rng()


    def _mask_instance(self, x_vec, mask):
        return (np.where(mask, x_vec, self.base_vec))[None, :]

    def explain(self, explicand: np.ndarray, num_samples: int) -> np.ndarray:
        x_vec = explicand.reshape(-1).astype(float)

        dummy_pred = self.model.predict(self._mask_instance(x_vec, np.zeros(self.n_feat, bool)))
        out_dim = 1 if np.asarray(dummy_pred).ndim == 1 else np.asarray(dummy_pred).shape[1]
        phi_acc = np.zeros((self.n_feat, out_dim), dtype=float)

        for _ in range(num_samples):
            perm = self.rng.permutation(self.n_feat)
            mask = np.zeros(self.n_feat, dtype=bool)

            prev_pred = self.model.predict(self._mask_instance(x_vec, mask))
            if out_dim == 1:
                prev_pred = prev_pred.reshape(1)

            for k, feat in enumerate(perm):
                mask[feat] = True
                curr_pred = self.model.predict(self._mask_instance(x_vec, mask))
                if out_dim == 1:
                    curr_pred = curr_pred.reshape(1)

                delta = curr_pred - prev_pred
                phi_acc[feat] += self.m_vec[k] * delta
                prev_pred = curr_pred

        phi = phi_acc / num_samples

        return phi[:, 0] if out_dim == 1 else phi

# ame_estimator.py
import numpy as np
import warnings

from ..base_estimator import BaseEstimator
from ..utils.p_generator import get_p

class _AMEBase(BaseEstimator):
    
    def __init__(self, model, baseline, weighting):
        super().__init__(model, baseline, weighting)
        self.base = np.asarray(baseline, float).reshape(-1)
        self.n = self.base.size

    @staticmethod
    def _extract_ab(weighting):
        parts = weighting.split("_")
        if len(parts) != 4:
            raise ValueError("beta_shapley string must be beta_shapley_α_β")
        return float(parts[-2]), float(parts[-1])

    def _masked(self, x_vec, mask):
        return np.where(mask, x_vec, self.base)[None, :]

    def _predict(self, X2d):
        pred = np.asarray(self.model.predict(X2d))
        if pred.ndim == 1:
            pred = pred[:, None]
        return pred.astype(float)

class AMEEstimator(_AMEBase):
    def __init__(
        self,
        model,
        baseline,
        weighting: str,
        *,
        eps_shapley: float = 1e-3,
        rng_seed: int | None = None,
    ):
        super().__init__(model, baseline, weighting)

        self.eps = eps_shapley
        self.rng = np.random.default_rng(rng_seed)

        p_vec = get_p(self.n, weighting)
        self.paired = np.allclose(p_vec, p_vec[::-1])

        if "banzhaf" in weighting:
            p = 0.5 if weighting == "banzhaf" else float(weighting.split("_")[-1])
            if not 0.0 < p < 1.0:
                raise ValueError("weighted_banzhaf needs 0<p<1 for AME.")
            self.sample_w = lambda p=p: p
            self.C = 1.0 / (p * (1.0 - p))

        elif weighting.startswith("beta_shapley"):
            α, β = self._extract_ab(weighting)
            if α <= 1 or β <= 1:
                raise ValueError("AME requires α>1, β>1 for Beta‑Shapley.")
            self.sample_w = lambda β=β, α=α: self.rng.beta(β, α)
            self.C = ((α + β - 1) * (α + β - 2)) / ((β - 1) * (α - 1))

        else:
            raise ValueError(
                "AME not implemented for this weighting: "
                "no finite‑C representation is known."
            )

    def explain(self, explicand, num_samples: int):
        """Return AME φ for a single instance."""
        x_vec = np.asarray(explicand, float).reshape(-1)
        if x_vec.size != self.n:
            raise ValueError("explicand and baseline have different n_features")

        dummy_pred = self._predict(self._masked(x_vec, np.zeros(self.n, bool)))
        out_dim = dummy_pred.shape[1]
        phi_acc = np.zeros((self.n, out_dim), dtype=float)

        for _ in range(num_samples):
            perm = self.rng.permutation(self.n)
            mask = np.zeros(self.n, dtype=bool)

            prev_pred = self._predict(self._masked(x_vec, mask))

            for k, feat in enumerate(perm):
                mask[feat] = True
                curr_pred = self._predict(self._masked(x_vec, mask))

                delta = curr_pred - prev_pred
                phi_acc[feat] += self.m_vec[k] * delta
                prev_pred = curr_pred

        phi = phi_acc / num_samples
        return phi[:, 0] if out_dim == 1 else phi

class ImprovedAMEEstimator(_AMEBase):
    """
    Improved Average‑Marginal‑Effect estimator (OFA, Prop 6)
    with automatic paired sampling for symmetric semivalues.
    """

    def __init__(
        self,
        model,
        baseline,
        weighting: str,
        *,
        eps_shapley: float = 1e-3,
        rng_seed: int | None = None,
    ):
        super().__init__(model, baseline, weighting)

        self.eps = eps_shapley
        self.rng = np.random.default_rng(rng_seed)

        p_vec = get_p(self.n, weighting)
        self.paired = np.allclose(p_vec, p_vec[::-1])

        if weighting in ("shapley", "beta_shapley_1_1"):
            eps = self.eps
            self.draw_w = lambda: self.rng.uniform(eps, 1.0 - eps)

        elif weighting.startswith("weighted_banzhaf"):
            p = 0.5 if weighting == "weighted_banzhaf" else float(weighting.split("_")[-1])
            if not (0.0 < p < 1.0):
                raise ValueError("weighted_banzhaf requires 0<p<1.")
            self.draw_w = lambda p=p: p        # Dirac at p

        elif weighting.startswith("beta_shapley"):
            alpha, beta = self._extract_ab(weighting)
            if alpha <= 1 or beta <= 1:
                raise ValueError("beta_shapley needs alpha>1, beta>1.")
            self.draw_w = lambda a=alpha, b=beta: self.rng.beta(b, a)

        else:
            raise ValueError("Improved AME not implemented for this weighting.")

    def explain(self, explicand, num_samples: int):
        x = np.asarray(explicand, float).reshape(-1)
        if x.size != self.n:
            raise ValueError("explicand length mismatch.")

        out_dim = self._predict(self.base[None, :]).shape[1]
        phi = np.zeros((self.n, out_dim))
        t = 0

        while t < num_samples:
            # sample first subset
            w = self.draw_w()
            mask = self.rng.random(self.n) < w
            U = self._predict(self._masked(x, mask))
            coeff = mask / w - (1 - mask) / (1 - w)
            phi += coeff[:, None] * U
            t += 1
            if not (self.paired and t < num_samples):
                continue  # no pairing or reached quota

            # paired complementary subset
            mask_c = ~mask
            U_c = self._predict(self._masked(x, mask_c))
            coeff_c = mask_c / w - (1 - mask_c) / (1 - w)
            phi += coeff_c[:, None] * U_c
            t += 1

        phi /= num_samples
        return phi[:, 0] if out_dim == 1 else phi
import numpy as np
from math import log

from ..base_estimator import BaseEstimator
from ..utils.p_generator import get_p

def _extract_ab(weighting):
    parts = weighting.split("_")
    if len(parts) != 4:
        raise ValueError("beta_shapley string must be beta_shapley_α_β")
    return float(parts[-2]), float(parts[-1])

class AMEEstimator(BaseEstimator):
    """
    AME estimator as defined in Lin et al. (2022), Figure 1.

    Parameters
    ----------
    eps : float, optional
        Required only for "shapley". 0 < eps < 0.5 .
    """

    def __init__(
        self,
        model,
        baseline,
        weighting: str,
        *,
        eps: float | None = None,
        rng_seed: int | None = None,
    ):
        super().__init__(model, baseline, weighting)

        self.model = model
        self.base = np.asarray(baseline, float).reshape(-1)
        self.n = self.base.size
        self.rng = np.random.default_rng(rng_seed)

        # choose μ and compute constant C
        if "banzhaf" in weighting:
            p = 0.5 if weighting == "banzhaf" else float(weighting.split("_")[-1])
            if not 0.0 < p < 1.0:
                raise ValueError("weighted_banzhaf requires 0<p<1")
            self.draw_w = lambda p=p: p           # Dirac mass δ_p
            self.C = 1.0 / (p * (1.0 - p))

        elif "beta_shapley" in weighting:
            alpha, beta = _extract_ab(weighting)
            if alpha <= 1 or beta <= 1:
                raise ValueError("Original AME needs alpha>1, beta>1.")
            # μ = Beta(beta, alpha)
            self.draw_w = lambda a=alpha, b=beta: self.rng.beta(b, a)
            self.C = ((alpha + beta - 1) * (alpha + beta - 2)) / (
                (beta - 1) * (alpha - 1)
            )

        elif "shapley" in weighting:
            if eps is None or not (0.0 < eps < 0.5):
                raise ValueError("Provide eps in (0,0.5) for shapley.")
            self.draw_w = lambda e=eps: self.rng.uniform(e, 1.0 - e)
            self.C = 2.0 * log((1.0 - eps) / eps) / (1.0 - 2.0 * eps)

        else:
            raise ValueError(
                "Original AME not applicable to this weighting "
                "(C would diverge). "
                "Use ImprovedAMEEstimator for accurate Shapley estimation."
            )

    def explain(self, explicand, num_samples: int):
        x = np.asarray(explicand, float).reshape(-1)

        p = get_p(self.n, self.weighting) 
        self.paired = np.allclose(p, p[::-1])
        
        if x.size != self.n:
            raise ValueError("explicand length mismatch.")

        # determine output dimension
        out_dim = self._predict(self.base[None, :]).shape[1]

        # accumulate AᵀA and Aᵀb incrementally
        ATA = np.zeros((self.n, self.n))
        ATb = np.zeros((self.n, out_dim))

        for _ in range(num_samples):
            w = self.draw_w()
            mask = self.rng.random(self.n) < w
            U = self._predict(self._masked(x, mask))  # shape (1,out_dim)

            X = mask / (w * self.C) - (1 - mask) / ((1 - w) * self.C)  # (n,)
            ATA += np.outer(X, X)
            ATb += X[:, None] * U

        # solve (ATA)φ = ATb
        phi = np.linalg.solve(ATA + 1e-12 * np.eye(self.n), ATb)  # tiny ridge

        return phi[:, 0] if out_dim == 1 else phi

    def _masked(self, x_vec, mask):
        return np.where(mask, x_vec, self.base)[None, :]

    def _predict(self, X2d):
        pred = np.asarray(self.model.predict(X2d))
        if pred.ndim == 1:
            pred = pred[:, None]
        return pred.astype(float)


class ImprovedAMEEstimator(BaseEstimator):
    """
    Improved Average‑Marginal‑Effect estimator (OFA, Proposition 6).
    """

    def __init__(
        self,
        model,
        baseline,
        weighting: str,
        *,
        rng_seed: int | None = None,
    ):
        super().__init__(model, baseline, weighting)

        self.base = np.asarray(baseline, float).reshape(-1)
        self.n = self.base.size
        self.rng = np.random.default_rng(rng_seed)

        if "banzhaf" in weighting:
            p = 0.5 if weighting == "banzhaf" else float(weighting.split("_")[-1])
            if not 0.0 < p < 1.0:
                raise ValueError("weighted_banzhaf requires 0<p<1.")
            self.draw_w = lambda p=p: p

        elif "beta_shapley" in weighting:
            alpha, beta = _extract_ab(weighting)               # returns (α, β)
            if alpha <= 1 or beta <= 1:
                raise ValueError("beta_shapley needs alpha>1, beta>1.")
            self.draw_w = lambda a=alpha, b=beta: self.rng.beta(b, a)

        elif "shapley" in weighting:
            self.draw_w = lambda: self.rng.random()            # U(0,1)
            # eps = 1e-3
            # self.draw_w = lambda e=eps: self.rng.uniform(e, 1.0 - e)

        else:
            raise ValueError("Improved AME not implemented for this weighting.")
        
        p_vec = get_p(self.n, weighting)
        self.paired = np.allclose(p_vec, p_vec[::-1])

    def explain(self, explicand, num_samples: int):
        x = np.asarray(explicand, float).reshape(-1)
        
        num_iters = num_samples // 2 if self.paired else num_samples

        set_phi_sum = False
        for _ in range(num_iters):
            w = self.draw_w()

            S = self.rng.random(self.n) < w                    # mask ∼ Ber(w)
            pred = self._predict(self._masked(x, S))

            if not set_phi_sum:
                out_dim = pred.shape[1]
                phi_sum = np.zeros((self.n, out_dim))
                set_phi_sum = True

            coeff = S / w - (1.0 - S) / (1.0 - w)              # Eq. (10)
            phi_sum += coeff[:, None] * pred

            if self.paired:
                Sc = ~S
                pred_c = self._predict(self._masked(x, Sc))
                coeff_c = Sc / w - (1.0 - Sc) / (1.0 - w)
                phi_sum += coeff_c[:, None] * pred_c

        phi = phi_sum / num_samples
        return phi[:, 0] if out_dim == 1 else phi

    def _masked(self, x_vec: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return np.where(mask, x_vec, self.base)[None, :]

    def _predict(self, X2d: np.ndarray) -> np.ndarray:
        y = np.asarray(self.model.predict(X2d))
        if y.ndim == 1:
            y = y[:, None]
        return y.astype(float)

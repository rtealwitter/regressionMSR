import numpy as np
from math import log

from ..base_estimator import BaseEstimator
from ..utils.p_generator import get_p, is_symmetric

def _extract_ab(weighting):
    parts = weighting.split("_")
    if len(parts) != 4:
        raise ValueError("beta_shapley string must be beta_shapley_α_β")
    return float(parts[-2]), float(parts[-1])

class AMEEstimator(BaseEstimator):
    """
    AME estimator (Lin et al., 2022, Fig. 1).

    Parameters
    ----------
    model      : fitted predictive model with .predict(X) -> ndarray
    baseline   : 1‑D array used to fill missing features
    weighting  : str  ("banzhaf", "weighted_banzhaf_p", "beta_shapley_a_b",
                       or "shapley")
    rng_seed   : int | None  – random seed for reproducibility
    eps_clip   : float       – ε‑truncation for the Shapley case (default 1e‑3)
    """

    def __init__(
        self,
        model,
        baseline,
        weighting: str,
        *,
        rng_seed: int | None = None,
        eps_clip: float = 1e-3,
    ):
        super().__init__(model, baseline, weighting)

        self.model   = model
        self.base    = np.asarray(baseline, float).reshape(-1)
        self.n       = self.base.size
        self.rng     = np.random.default_rng(rng_seed)
        self.eps     = eps_clip

        # choose μ(p) and compute C = E[1/(p(1-p))]
        if "banzhaf" in weighting:
            p = 0.5 if weighting == "banzhaf" else float(weighting.split("_")[-1])
            if not 0.0 < p < 1.0:
                raise ValueError("weighted_banzhaf requires 0<p<1.")
            self.draw_w = lambda p=p: p
            self.C      = 1.0 / (p * (1.0 - p))

        elif weighting.startswith("beta_shapley"):
            alpha, beta = _extract_ab(weighting)
            if alpha <= 1 or beta <= 1:
                raise ValueError("beta_shapley needs alpha>1 and beta>1.")
            self.draw_w = lambda a=alpha, b=beta: self.rng.beta(a, b)
            self.C      = ((alpha + beta - 1) * (alpha + beta - 2)) / (
                (alpha - 1) * (beta - 1)
            )

        elif weighting == "shapley":
            # ε‑truncated uniform
            eps = self.eps
            self.draw_w = lambda e=eps: self.rng.uniform(e, 1.0 - e)
            self.C      = 2.0 * log((1.0 - eps) / eps) / (1.0 - 2.0 * eps)

        else:
            raise ValueError(
                "AME supports banzhaf, weighted_banzhaf_p, beta_shapley_a_b, "
                "or shapley."
            )
        p_vec = get_p(self.n, weighting)
        self.paired = is_symmetric(p_vec)

    def explain(self, explicand, num_samples: int):
        x = np.asarray(explicand, float).reshape(-1)

        out_dim = self._predict(self.base[None, :]).shape[1]
        ATA = np.zeros((self.n, self.n))
        ATb = np.zeros((self.n, out_dim))

        loops = num_samples // 2 if self.paired else num_samples

        for _ in range(loops):
            w = self.draw_w()

            mask = self.rng.random(self.n) < w
            U = self._predict(self._masked(x, mask))
            X = mask / (w * self.C) - (1.0 - mask) / ((1.0 - w) * self.C)
            ATA += np.outer(X, X)
            ATb += X[:, None] * U

            if self.paired:
                mask_c = ~mask
                U_c    = self._predict(self._masked(x, mask_c))
                X_c    = mask_c / (w * self.C) - (1.0 - mask_c) / ((1.0 - w) * self.C)
                ATA += np.outer(X_c, X_c)
                ATb += X_c[:, None] * U_c

        # solve (AᵀA) φ = Aᵀb  (tiny ridge for stability)
        phi = np.linalg.solve(ATA * np.eye(self.n), ATb)
        return phi[:, 0] if out_dim == 1 else phi

    def _masked(self, x_vec, mask):
        return np.where(mask, x_vec, self.base)[None, :]

    def _predict(self, X2d):
        y = np.asarray(self.model.predict(X2d))
        return y[:, None] if y.ndim == 1 else y.astype(float)


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

        if weighting == "beta_shapley_1_1":
            weighting = "shapley"

        if "banzhaf" in weighting:
            p = 0.5 if weighting == "banzhaf" else float(weighting.split("_")[-1])
            if not 0.0 < p < 1.0:
                raise ValueError("weighted_banzhaf requires 0<p<1.")
            self.draw_w = lambda p=p: p

        elif "beta_shapley" in weighting:
            alpha, beta = _extract_ab(weighting)               # returns (α, β)
            self.draw_w = lambda a=alpha, b=beta: self.rng.beta(b, a)

        elif "shapley" in weighting:
            self.draw_w = lambda: self.rng.random()            # U(0,1)
            # eps = 1e-3
            # self.draw_w = lambda e=eps: self.rng.uniform(e, 1.0 - e)

        else:
            raise ValueError("Improved AME not implemented for this weighting.")
        
        p_vec = get_p(self.n, weighting)
        self.paired = is_symmetric(p_vec)

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

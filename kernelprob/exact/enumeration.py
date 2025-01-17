import itertools
import numpy as np
import warnings

from ..p_generator import get_p

def enumeration_prob(baseline, explicands, model, weighting="shapley"):
    warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in scalar multiply")
    true_values = []
    explainer = EnumerationExplainer(model=model, baseline=baseline, weighting=weighting)
    for explicand in explicands:
        true_values.append(explainer.explain(explicand))
    return true_values


class EnumerationExplainer:
    """
    Exponential-time explainer that enumerates every subset of features
    and computes exact attributions according to the weighting distribution
    """
    def __init__(self, model, baseline, weighting="shapley"):
        self.model = model
        baseline = np.array(baseline, dtype=float).ravel()
        self.baseline = baseline
        self.weighting = weighting

    def _build_subsets_in_batches(self, explicand, batch_size=4096):
        N = len(explicand)
        n_subsets = 1 << N

        baseline_copy = self.baseline

        for start_idx in range(0, n_subsets, batch_size):
            end_idx = min(start_idx + batch_size, n_subsets)
            chunk_size = end_idx - start_idx

            X_chunk = np.tile(baseline_copy, (chunk_size, 1))
            subsets_chunk = []

            for offset in range(chunk_size):
                subset_index = start_idx + offset
                subset = []
                for feat in range(N):
                    if (subset_index & (1 << feat)) != 0:
                        subset.append(feat)
                        X_chunk[offset, feat] = explicand[feat]
                subsets_chunk.append(frozenset(subset))

            yield subsets_chunk, X_chunk

    def explain(self, explicand, batch_size=4096):
        import warnings
        warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")

        explicand = np.array(explicand, dtype=float).ravel()
        N = explicand.shape[0]

        p = get_p(N, self.weighting)

        f_cache = {}
        for subsets_chunk, X_chunk in self._build_subsets_in_batches(explicand, batch_size):
            predictions = self.model.predict(X_chunk) 
            for s_key, pred in zip(subsets_chunk, predictions):
                f_cache[s_key] = pred

        values = np.zeros(N, dtype=float)
        all_features = range(N)

        for i in range(N):
            exclude_i = [f for f in all_features if f != i]
            for r in range(N + 1):
                for subset in itertools.combinations(exclude_i, r):
                    s_key = frozenset(subset)
                    s_union_i_key = s_key.union({i})
                    delta = f_cache[s_union_i_key] - f_cache[s_key]
                    values[i] += delta * p[r]

        return values
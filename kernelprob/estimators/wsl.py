import numpy as np
from typing import Optional, Union, Tuple
from scipy import special

from ..base_estimator import BaseEstimator


class WSLEstimator(BaseEstimator):
    """
    A unified Weighted Semivalue Lift (WSL)-style estimator that handles
    various weighting schemes (shapley, banzhaf, beta_shapley, weighted_banzhaf).
    Paired sampling is always used.
    
    Additional Parameters
    nue_avg : int
        Total number of samples to draw for the estimator.
    nue_track_avg : int, default=10
        Number of samples to aggregate before updating the trajectory.
    nue_per_proc : int
        Number of samples per processing batch (used internally).
    """

    def __init__(
        self,
        model,
        baseline: np.ndarray,
        weighting: str,
    ):
        super().__init__(model, baseline, weighting)

        self.semivalue, self.semivalue_param = self._parse_weighting(weighting)

        # Aggregator / sampling logic parameters
        self.nue_avg = 0        # Total number of samples
        self.nue_track_avg = 10
        self.nue_per_proc = 1000

        # Number of players (features)
        self.num_player = self.baseline.size

        # Calculate derived parameters
        self.interval_track = self.nue_track_avg // 2
        self.num_sample = self.nue_avg // 2               # Paired sampling: half the samples
        self.batch_size = -(-self.nue_per_proc // (2 * self.num_player))  # Ceiling division

        # Initialize aggregation arrays
        self.results_aggregate = {
            "estimates": np.zeros(self.num_player, dtype=np.float64),
            "count": 0
        }

        # Initialize buffers
        self.buffer_size = self.interval_track + self.batch_size - 1
        self.buffer = np.empty((self.buffer_size, self.num_player), dtype=np.float64)
        self.samples = np.empty((self.batch_size, self.num_player - 1), dtype=bool)

        # Paired sampling: always set lock_switch to False to allow toggling
        self.lock_switch = False
        self.switch_state = False  # Initial switch state

        # Initialize trajectory tracking (optional, can be removed if not needed)
        self.pos_buffer = 0
        self.pos_traj = 0
        self.values_traj = np.empty((self.nue_avg // self.nue_track_avg, self.num_player), dtype=np.float64)

        # Precompute "WSL" distribution weights if not shapley
        self.is_wsl = (self.semivalue != "shapley")

        # Multiply by num_player if it's WSL
        vs = self.distribution_cardinality()
        if self.is_wsl:
            self.weights = vs * self.num_player
        else:
            self.weights = None  # Shapley does not require weighting

    def distribution_cardinality(self) -> np.ndarray:
        """
        Compute semivalue distribution weights for each possible subset size
        based on the semivalue type.
        
        Returns
        -------
        weights : np.ndarray
            An array of length `num_player` containing the distribution weights.
        """
        n = self.num_player
        if self.semivalue == "shapley":
            # Each of n players has equal weight
            weights = np.full(n, 1.0 / n, dtype=np.float64)

        elif self.semivalue == "weighted_banzhaf":
            # Weighted Banzhaf with parameter t
            t = float(self.semivalue_param)
            weights = np.ones(n, dtype=np.float64)
            for k in range(n):
                for i in range(k):
                    weights[k] *= ((n - 1 - i) / (i + 1)) * t * (1 - t)
                ex = (n - 1 - 2 * k)
                if ex < 0:
                    weights[k] *= (1 - t) ** ex
                else:
                    weights[k] *= (1 - t) ** ex

        elif self.semivalue == "beta_shapley":
            # Beta Shapley with parameters alpha and beta
            alpha, beta_ = self.semivalue_param
            weights = np.zeros(n, dtype=np.float64)
            for s in range(n):
                r_cur = 1.0
                if s > 0:
                    rng1 = np.arange(1, s + 1, dtype=np.float64)
                    r_cur *= np.prod((rng1 + (beta_ - 1.0)) / rng1)
                if (n - s) > 1:
                    rng2 = np.arange(1, n - s, dtype=np.float64)
                    r_cur *= np.prod(((alpha - 1.0) + rng2) / rng2)
                weights[s] = r_cur

        else:
            raise NotImplementedError(f"Check {self.semivalue}")

        return weights

    def _parse_weighting(self, weighting_str: str) -> Tuple[str, Optional[Union[float, Tuple[float, float]]]]:
        if weighting_str == "shapley":
            return ("shapley", None)
        elif weighting_str == "banzhaf":
            return ("weighted_banzhaf", 0.5)
        elif weighting_str.startswith("beta_shapley_"):
            # e.g., "beta_shapley_2_5"
            parts = weighting_str.split("_")
            if len(parts) != 4:
                raise ValueError(f"Invalid beta_shapley format: '{weighting_str}', expected 'beta_shapley_alpha_beta'.")
            alpha = float(parts[2])
            beta = float(parts[3])
            return ("beta_shapley", (alpha, beta))
        elif weighting_str.startswith("weighted_banzhaf_"):
            # e.g., "weighted_banzhaf_0.3"
            parts = weighting_str.split("_")
            if len(parts) != 3:
                raise ValueError(f"Invalid weighted_banzhaf format: '{weighting_str}', expected 'weighted_banzhaf_p'.")
            p = float(parts[2])
            return ("weighted_banzhaf", p)
        else:
            raise ValueError(f"Unknown weighting: '{weighting_str}'")

    # Aggregator properties and methods
    @property
    def switch(self) -> bool:
        return self.switch_state

    @switch.setter
    def switch(self, state: bool):
        if not self.lock_switch:
            self.switch_state = state

    def _init_indiv(self):
        """
        Initialize individual-specific settings before sampling.
        """
        if self.semivalue == "beta_shapley":
            alpha, beta_ = self.semivalue_param

    # Sampling generator (always paired)
    def sampling(self):
        """
        Generator that yields batches of sampled subsets using paired sampling.
        Each subset is complemented by its inverse to reduce variance.
        
        Yields
        ------
        np.ndarray
            A batch of sampled subsets of shape (batch_size, num_player - 1).
        """
        self._init_indiv()

        count = 0
        for _ in range(self.num_sample):
            if not self.switch:
                # Normal draw
                self.samples[count] = self._generator()
                self.switch = True
            else:
                # Paired draw: complement of the previous subset
                self.samples[count] = ~self.samples[count - 1]
                self.switch = False

            count += 1
            if count == self.batch_size:
                yield self.samples.copy()
                count = 0

        if count:
            yield self.samples[:count]

    # Subset generator based on semivalue
    def _generator(self) -> np.ndarray:
        """
        Generate a single subset based on the semivalue weighting.
        
        Returns
        -------
        np.ndarray
            A boolean array of shape (num_player - 1,) indicating feature inclusion.
        """
        if self.semivalue == "shapley":
            t = np.random.rand()
        elif self.semivalue == "weighted_banzhaf":
            t = float(self.semivalue_param)
        elif self.semivalue == "beta_shapley":
            alpha, beta_ = self.semivalue_param
            # Note: The original snippet used (beta_, alpha) in np.random.beta
            t = np.random.beta(beta_, alpha)
        elif self.semivalue == "banzhaf":
            # WSL_banzhaf uses fixed p=0.5
            t = 0.5
        else:
            raise NotImplementedError(f"Unsupported semivalue: {self.semivalue}")

        return np.random.binomial(1, t, size=self.num_player - 1).astype(bool)

    # Run method: evaluates the model on sampled subsets
    def run(self, samples: np.ndarray) -> np.ndarray:
        """
        Evaluate the model for each subset in `samples` and compute the attribution differences.
        
        Parameters
        ----------
        samples : np.ndarray
            A batch of sampled subsets of shape (batch_size, num_player - 1).
        
        Returns
        -------
        np.ndarray
            An array of shape (batch_size, num_player) containing the attribution differences.
        """
        results_collect = np.zeros((len(samples), self.num_player), dtype=np.float64)
        subset = np.zeros(self.num_player, dtype=bool)

        for i, sample in enumerate(samples):
            results = results_collect[i]
            subset[:self.num_player - 1] = sample

            # results[-1] -= evaluate(subset) with subset[-1] = 0
            val0 = self._evaluate_model(subset)
            results[-1] -= val0

            # subset[-1] = 1
            subset[-1] = True
            val1 = self._evaluate_model(subset)
            results[-1] += val1

            # Iterate over each player to compute individual attributions
            for player in range(self.num_player - 1):
                # Swap subset[-1] and subset[player]
                tmp = subset[-1]
                subset[-1] = subset[player]
                subset[player] = tmp

                # results[player] += evaluate(subset)
                val_plus = self._evaluate_model(subset)

                # subset[player] = 0
                subset[player] = False
                val_minus = self._evaluate_model(subset)
                results[player] += val_plus
                results[player] -= val_minus

                # Revert the swap
                subset[player] = tmp
                subset[-1] = tmp

            # Revert subset[-1] to 0
            subset[-1] = False

        # If WSL, multiply by distribution weights based on subset sizes
        if self.is_wsl:
            subset_sizes = samples.sum(axis=1)
            scalars = self.weights[subset_sizes]
            results_collect = scalars[:, None] * results_collect

        return results_collect

    # Evaluate the model given a subset
    def _evaluate_model(self, subset_bool: np.ndarray) -> float:
        """
        Evaluate the model on the perturbed input defined by `subset_bool`.
        
        Parameters
        ----------
        subset_bool : np.ndarray
            A boolean array of shape (num_player,) indicating feature inclusion.
        
        Returns
        -------
        float
            The scalar prediction from the model.
        """
        # Create the perturbed input: features included are from explicand, others from baseline
        x = np.where(subset_bool, self._explicand, self.baseline)
        val = self.model.predict(x.reshape(1, -1))
        return float(val[0])  # Assuming scalar output

    # Aggregate results
    def aggregate(self, results_collect: np.ndarray):
        """
        Aggregate the results by performing a weighted running average.
        
        Parameters
        ----------
        results_collect : np.ndarray
            An array of shape (batch_size, num_player) containing attribution differences.
        """
        num_pre = self.results_aggregate["count"]
        num_cur = num_pre + len(results_collect)

        # Weighted running average
        self.results_aggregate["estimates"] *= (num_pre / num_cur)
        self.results_aggregate["estimates"] += (results_collect.sum(axis=0) / num_cur)
        self.results_aggregate["count"] = num_cur

    # Finalize and return the aggregated attributions
    def finalize(self) -> np.ndarray:
        """
        Finalize the aggregation and return the final attributions.
        
        Returns
        -------
        np.ndarray
            The final attribution array of shape (num_player,).
        """
        return self.results_aggregate["estimates"].copy()


    def explain(self, explicand: np.ndarray, num_samples: int) -> np.ndarray:
        """
        Compute the attributions/feature importances for the given explicand.
        
        Parameters
        ----------
        explicand : np.ndarray
            The instance to be explained, of shape (num_player,).
        num_samples : int
            Total number of samples to draw for the estimator.
        
        Returns
        -------
        phi : np.ndarray
            The final attributions for each feature, of shape (num_player,).
        """
        self._explicand = np.asarray(explicand, dtype=float)

        # Update sampling parameters based on num_samples
        self.nue_avg = num_samples // self.num_player
        self.num_sample = self.nue_avg // 2  # Paired sampling: half the samples

        # Reset aggregator
        self.results_aggregate["estimates"][:] = 0.0
        self.results_aggregate["count"] = 0

        for batch in self.sampling():
            results = self.run(batch)
            self.aggregate(results)

        return self.finalize()
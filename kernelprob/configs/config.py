from typing import Any, Dict, List
import itertools
import yaml

# from ..estimators.kpc import KernelProbCEstimator
# from ..estimators.kpu import KernelProbUEstimator
from ..estimators.reg import UniversalConstrained, UniversalConstrainedPair, UniversalUnconstrained, UniversalUnconstrainedPair

from ..estimators.ofa import OFA_S, OFA_A
from ..estimators.wsl import WSLEstimator

from ..estimators.lshap import LeverageSHAPEstimator
from ..estimators.kshap import KernelSHAPEstimator
from ..estimators.pshap import PermutationSHAPEstimator
from ..estimators.kb import KernelBanzhafEstimator
from ..estimators.msr import MSREstimator
from ..estimators.mc import MonteCarloEstimator

from ..exact.treeprob import TreeProbExplainer
from ..exact.enumeration import EnumerationExplainer

config_path = "kernelprob/configs/estimators.yaml"

ESTIMATOR_TRUE = {
    "TreeProb": TreeProbExplainer,
    "Enumeration": EnumerationExplainer,
}

ESTIMATOR_ALL = {
    "UniversalConstrained": UniversalConstrained,
    "UniversalConstrainedPair": UniversalConstrainedPair,
    "UniversalUnconstrained": UniversalUnconstrained,
    "UniversalUnconstrainedPair": UniversalUnconstrainedPair,
    "OFA_S": OFA_S,
    "OFA_A": OFA_A,
    "WSL": WSLEstimator,
}

ESTIMATOR_SHAPLEY = {
    "LeverageSHAP": LeverageSHAPEstimator,
    "KernelSHAP": KernelSHAPEstimator,
    "PermutationSHAP": PermutationSHAPEstimator,
}

ESTIMATOR_BANZHAF = {
    #"KernelBanzhaf": KernelBanzhafEstimator,
    "MSR": MSREstimator,
    "MonteCarlo": MonteCarloEstimator,
}

# Combine all estimator dictionaries
ESTIMATOR_FACTORY = {**ESTIMATOR_ALL, **ESTIMATOR_SHAPLEY, **ESTIMATOR_BANZHAF}

DATASETS = {
    "small_n": ['California', 'Adult'],
    "big_n": ['Correlated', 'Independent', 'NHANES', 'Communities']
}

VARIABLES = {
    "sample_size":
        {
            "sample_size": [6, 10, 20, 40, 80, 160],
            "noise_std": [0]
        },
    "noise_std":
        {
            "sample_size": [10],
            "noise_std": [0, .5 * 1e-3, 1e-3, .5 * 1e-2, 1e-2, .5 * 1e-1, 1e-1, .5, 1]
        }
    # "bn": TODO
}

NUM_RUNS = 10

def generate_experiment_configs() -> List[Dict[str, Any]]:
    """
    Generate a list of experiment configurations by creating the Cartesian product
    of parameters for each estimator.
    """
    config = load_config(config_path)
    experiments = []
    
    estimators_config = config.get("estimators", {})
    weightings_config = config.get("weightings", {})

    for estimator_name, params in estimators_config.items():
        weighting_categories = params.get("weightings", [])

        applicable_weightings = []
        for category in weighting_categories:
            category_weightings = weightings_config.get(category, [])
            applicable_weightings.extend(category_weightings)

        for weighting in applicable_weightings:
            experiments.append({
                "name": f"{estimator_name}_{weighting}",
                "estimator_class": estimator_name,
                "weighting": weighting,
            })

    return experiments

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config
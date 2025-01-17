from typing import Any, Dict, List
import itertools
import yaml

from ..estimators.kpc import KernelProbCEstimator
from ..estimators.kpu import KernelProbUEstimator

from ..estimators.ofa import OFAEstimator
from ..estimators.wsl import WSLEstimator

from ..estimators.lshap import LeverageSHAPEstimator
from ..estimators.kshap import KernelSHAPEstimator
from ..estimators.pshap import PermutationSHAPEstimator
from ..estimators.kb import KernelBanzhafEstimator
from ..estimators.mc import MonteCarloEstimator

from ..exact.treeprob import TreeProbExplainer
from ..exact.enumeration import EnumerationExplainer

config_path = "kernelprob/configs/estimators.yaml"

ESTIMATOR_FACTORY = {
    # ground truth
    "TreeProb": TreeProbExplainer,
    "Enumeration": EnumerationExplainer, # For random p distribution
    # estimators
    "KernelProbC": KernelProbCEstimator,
    "KernelProbU": KernelProbUEstimator,
    "OFA": OFAEstimator, # OFA is the general version of MSR
    "WSL": WSLEstimator, # WSL is the general version of MC
    # ARM has "worst" performance in OFA so we exclude it
    # standard only
    "LeverageSHAP": LeverageSHAPEstimator,
    "KernelSHAP": KernelSHAPEstimator,
    "PermutationSHAP": PermutationSHAPEstimator,
    "KernelBanzhaf": KernelBanzhafEstimator,
    "MonteCarlo": MonteCarloEstimator, 
}

DATASETS = {
    "small_n": ['IRIS', 'California', 'Adult', 'Breast Cancer'],
    "big_n": ['Correlated', 'Independent', 'NHANES', 'Communities']
}

VARIABLES = {
    "sample_size":
        {
            "sample_size": [5, 10, 20, 40, 80, 160],
            "noise_std": [0]
        },
    "noise_std":
        {
            "sample_size": [10],
            "noise_std": [0, .5 * 1e-3, 1e-3, .5 * 1e-2, 1e-2, .5 * 1e-1, 1e-1, .5, 1]
        }
}

NUM_RUNS = 5

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

        if "modes" in params: # OFAEstimator
            modes = params.get("modes", [])
            # Create Cartesian product for OFAEstimator
            for weighting, mode in itertools.product(applicable_weightings, modes):
                experiments.append({
                    "name": f"{estimator_name}_{mode}_{weighting}",
                    "estimator_class": estimator_name,
                    "weighting": weighting,
                    "mode": mode,
                })
        elif "paired_sampling" in params:
            paired_sampling = params.get("paired_sampling", [])
            leverage_sampling = params.get("leverage_sampling", [])
            bernoulli_sampling = params.get("bernoulli_sampling", [])
            for weighting, paired, leverage, bernoulli in itertools.product(applicable_weightings, paired_sampling, leverage_sampling, bernoulli_sampling):
                experiments.append({
                    "name": f"{estimator_name}_{weighting}_paired-{paired}_leverage-{leverage}_bernoulli-{bernoulli}",
                    "estimator_class": estimator_name,
                    "weighting": weighting,
                    "paired_sampling": paired,
                    "leverage_sampling": leverage,
                    "bernoulli_sampling": bernoulli,
                })
        else:
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
from typing import Any, Dict, List
import itertools
import yaml

from ..estimators.kpc import KernelProbCEstimator
from ..estimators.kpu import KernelProbUEstimator
from ..estimators.lshap import LeverageSHAPEstimator
from ..estimators.mc import MonteCarloEstimator
from ..estimators.kb import KernelBanzhafEstimator

from ..estimators.ofa import OFAEstimator
from ..estimators.wsl import WSLEstimator

from ..gt.treeprob import TreeProbExplainer
from ..gt.enumeration import EnumerationExplainer

def generate_experiment_configs(variable: str) -> List[Dict[str, Any]]:
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
        modes = params.get("modes", []) if estimator_name == "OFAEstimator" else [None]

        applicable_weightings = []
        for category in weighting_categories:
            category_weightings = weightings_config.get(category, [])
            applicable_weightings.extend(category_weightings)

        if estimator_name == "OFAEstimator":
            # Create Cartesian product for OFAEstimator
            for weighting, mode in itertools.product(applicable_weightings, modes):
                experiments.append({
                    "name": f"{estimator_name}_mode{mode}_{weighting}",
                    "estimator_class": estimator_name,
                    "weighting": weighting,
                    "mode": mode,
                })
        else:
            for weighting in itertools.product(applicable_weightings):
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

config_path = "estimators.yml"

ESTIMATOR_FACTORY = {
    # ground truth
    "TreeProb": TreeProbExplainer,
    # estimators
    "KernelProbC": KernelProbCEstimator,
    "KernelProbU": KernelProbUEstimator,
    "OFA": OFAEstimator, # OFA is the general version of MSR
    "WSL": WSLEstimator, # WSL is the general version of MC
    # ARM has "worst" performance in OFA so we exclude it
    # standard only
    "LeverageSHAP": LeverageSHAPEstimator,
    "KernelBanzhaf": KernelBanzhafEstimator,
    "MonteCarlo": MonteCarloEstimator, 
}

DATASETS = {
    "small_n": ['IRIS', 'California', 'Diabetes', 'Adult'],
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
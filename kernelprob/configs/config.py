from typing import Any, Dict, List
import yaml

# import from utils
from ..utils.datasets import dataset_loaders, dataset_sizes
# from ..estimators.kpc import KernelProbCEstimator
# from ..estimators.kpu import KernelProbUEstimator
from ..estimators.reg import UniversalConstrained, UniversalUnconstrained
from ..estimators.regMSR import TreeMSR, LinearMSR, MSR

from ..estimators.ofa import OFA_S, OFA_A
from ..estimators.wsl import WSLEstimator
from ..estimators.gels import GELSEstimator, GELSShapleyEstimator
from ..estimators.arm import ARMEstimator
from ..estimators.shap_iq import SHAPIQEstimator
from ..estimators.wshap import WeightedSHAPEstimator
from ..estimators.ame import AMEEstimator, ImprovedAMEEstimator

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

# Important that supersets of other estimators are included first
# E.g., OFA_S_Optimized should be included before OFA_S
ESTIMATOR_ALL = {
    'TreeMSR': TreeMSR,
    'LinearMSR': LinearMSR,
    "OFA_S": OFA_S,
    "WSL": WSLEstimator,
    "GELS": GELSEstimator,
    "ARM": ARMEstimator,
    "SHAPIQ": SHAPIQEstimator,
    "WeightedSHAP": WeightedSHAPEstimator,
    "AME": AMEEstimator,
    "ImprovedAME": ImprovedAMEEstimator,
#    "OFA_A": OFA_A, # Unfair comparison
#    "LeverageSHAP+": UniversalConstrained,
#    "KernelBanzhaf+": UniversalUnconstrained,
}

ESTIMATOR_SHAPLEY = {
    "LeverageSHAP": LeverageSHAPEstimator,
    "KernelSHAP": KernelSHAPEstimator,
    "PermutationSHAP": PermutationSHAPEstimator,
    'MSR' : MSR,
    "GELS_S": GELSShapleyEstimator,
}

ESTIMATOR_BANZHAF = {
    "KernelBanzhaf": KernelBanzhafEstimator,
#    "MSR": MSREstimator,
    "BanzhafMonteCarlo": MonteCarloEstimator,
}

# Combine all estimator dictionaries
ESTIMATOR_FACTORY = {**ESTIMATOR_ALL, **ESTIMATOR_SHAPLEY, **ESTIMATOR_BANZHAF}

DATASETS = {'small_n' : [], 'big_n' : []}

for dataset in dataset_loaders:
    if dataset_sizes[dataset] < 20:
        DATASETS['small_n'].append(dataset)
    else:
        DATASETS['big_n'].append(dataset)

VARIABLES = {
    "basic" : {
        "sample_size": [10],
        "noise_std": [0],
    },
    "sample_size":
        {
            "sample_size": [6, 10, 20, 40, 80, 160],
            "noise_std": [0]
        },
    "noise_std":
        {
            "sample_size": [10],
            "noise_std": [.5 * 1e-3, 1e-3, .5 * 1e-2, 1e-2, .5 * 1e-1, 1e-1, .5, 1]
        }
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
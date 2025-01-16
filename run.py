import numpy as np
from typing import Any, Dict
from sklearn.linear_model import LogisticRegression
import argparse
import logging
import sklearn.ensemble

from kernelprob.configs.config import generate_experiment_configs, ESTIMATOR_FACTORY, DATASETS, VARIABLES, NUM_RUNS
from utils.datasets import load_dataset, load_input
from yurong.utils.run_utils import *
from kernelprob.gt.treeprob import tree_prob

def run_experiment(exp_cfg: Dict[str, Any], baseline, explicands, model: Any, variable: str, dataset: str, true_values: np.ndarray):
    estimator_class_name = exp_cfg["estimator_class"]
    estimator_cls = ESTIMATOR_FACTORY[estimator_class_name]
    if estimator_cls is None:
        logging.error(f"Estimator class '{estimator_class_name}' not found. Skipping.")
        return
    
    weighting = exp_cfg.get("weighting", "shapley")
    true_values = tree_prob(baseline, explicands, model, weighting)

    for noise_std in VARIABLES[variable][noise_std]:
        noised_model = NoisyModel(model, noise_std)
        try:
            if estimator_class_name == "OFA":
                mode = exp_cfg.get("mode", "A")
                estimator = estimator_cls(
                    model=noised_model,
                    baseline=baseline,
                    weighting=weighting,
                    mode=mode
                )
            else:
                estimator = estimator_cls(
                    model=noised_model,
                    baseline=baseline,
                    weighting=weighting
                )
        except Exception as e:
            logging.error(f"Error instantiating '{estimator_class_name}': {e}. Skipping.")
            return
        
        for sample_size in VARIABLES[variable][sample_size]:
            for idx, explicand in enumerate(explicands):
            # for bn in VARIABLES[variable]["bn"]:
                predicted = estimator.explain(explicand, sample_size)

                filename = get_filename(dataset, estimator_class_name, weighting)
                if not os.path.exists(filename):
                    with open(filename, 'w') as f:
                        f.write('')

                with open(filename, 'a') as f:
                    dict = {
                        'sample_size': sample_size,
                        'difference': noised_model.get_sample_count() - sample_size,
                        'noise': noise_std,
                        'n' : explicand.shape[0],
                        'error' : ((predicted - true_values[idx]) ** 2).mean(),
                    }
                    f.write(str(dict) + '\n')


def run_debug():
    dataset = load_dataset("California")
    X, y = dataset
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=8)
    model.fit(X, y)
    baseline, explicands = load_input(X, num_runs=1, is_synthetic=False)

    # ofa_a_estimator = ESTIMATOR_FACTORY["OFA"](model, baseline, weighting="banzhaf", mode="A")
    ofa_s_estimator = ESTIMATOR_FACTORY["OFA"](model, baseline, weighting="shapley", mode="S")
    # kernel_banzhaf_estimator = ESTIMATOR_FACTORY["KernelBanzhaf"](model, baseline, weighting="banzhaf")
    wsl_estimator = ESTIMATOR_FACTORY["WSL"](model, baseline, weighting="shapley")

    true_values = tree_prob(baseline, explicands, model, "shapley")
    print(f"true_values: {true_values}")

    for sample_size in [5, 10, 20]:
        for explicand in explicands:
            # ofa_a_phi = ofa_a_estimator.explain(explicand, sample_size)
            # print(f"ofa_a l2_error: {((ofa_a_phi - true_values) ** 2).mean()}")
            ofa_s_phi = ofa_s_estimator.explain(explicand, sample_size)
            print(f"ofa_s l2_error: {((ofa_s_phi - true_values) ** 2).mean()}")
            # kernel_banzhaf_phi = kernel_banzhaf_estimator.explain(explicand, sample_size)
            # print(f"kernel_banzhaf l2_error: {((kernel_banzhaf_phi - true_values) ** 2).mean()}")
            wsl_phi = wsl_estimator.explain(explicand, sample_size)
            print(f"wsl l2_error: {((wsl_phi - true_values) ** 2).mean()}")


    
def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument(
       "--variable",
        type=str,
        default="sample_size",
        help="sample_size, noise, or bn"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    return parser.parse_args()

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def main():
    setup_logging()

    args = parse_args()
    variable, debug = args.variable, args.debug

    if debug:
        run_debug()
        exit()

    experiments = generate_experiment_configs(variable)
    datasets = DATASETS["small_n"] + DATASETS["big_n"]

    for dataset in datasets:
        X, y = load_dataset(dataset)
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=8)
        model.fit(X, y)
        baseline, explicands = load_input(X, NUM_RUNS, is_synthetic=False)
        for exp_cfg in experiments:
            run_experiment(exp_cfg, baseline, explicands, model, variable, dataset)

if __name__ == "__main__":
    main()
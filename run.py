from typing import Any, Dict
import argparse
import logging
import sklearn.ensemble
import sklearn.neural_network
import sklearn.datasets

from kernelprob.configs.config import generate_experiment_configs, ESTIMATOR_FACTORY, DATASETS, VARIABLES, NUM_RUNS
from kernelprob.utils.datasets import load_dataset, load_input
from kernelprob.utils.p_generator import get_p
from kernelprob.utils.run_utils import *
from kernelprob.exact.treeprob import tree_prob
from kernelprob.exact.treeshap import tree_shap
from kernelprob.exact.enumeration import enumeration_prob

def run_experiment(exp_cfg: Dict[str, Any], baseline, explicands, model: Any, variable: str, dataset: str):
    n = baseline.shape[1]
    estimator_class_name = exp_cfg["estimator_class"]
    experiment_name = exp_cfg["name"]
    estimator_cls = ESTIMATOR_FACTORY[estimator_class_name]
    if estimator_cls is None:
        logging.error(f"Estimator class '{estimator_class_name}' not found. Skipping.")
        return
    
    weighting = exp_cfg.get("weighting", None)
    if weighting is None:
        logging.error(f"Weighting not specified for '{experiment_name}'. Skipping.")

    if "random" in weighting:
        logging.info("Using enumeration for random p distribution.")
        true_values = enumeration_prob(baseline, explicands, model, weighting)
    # TODO: remove if treeprob is good enough :)
    elif weighting == "shapley":
        true_values = tree_shap(baseline, explicands, model)
    else:
        true_values = tree_prob(baseline, explicands, model, weighting)
    
    # # Discrepancy check
    # official_tree_shap = tree_shap(baseline, explicands, model)
    # for idx, true_value in enumerate(true_values):
    #     if np.allclose(true_value, official_tree_shap[idx], rtol=1e-6, atol=1e-8):
    #         print("Agrees with official tree SHAP.")
    #     else:
    #         print("Does not agree with official tree SHAP.")
    #         print(f"True value: {true_value}")
    #         print(f"Official tree SHAP: {official_tree_shap[idx]}")
    # exit()

    for noise_std in VARIABLES[variable]["noise_std"]:
        noised_model = NoisyModel(model, noise_std)
        try:
            estimator = estimator_cls(
                model=noised_model,
                baseline=baseline,
                weighting=weighting
            )
        except Exception as e:
            logging.error(f"Error instantiating '{estimator_class_name}': {e}. Skipping.")
            return
        
        for sample_size in VARIABLES[variable]["sample_size"]:
            actual_sample_size = sample_size * n
            for idx, explicand in enumerate(explicands):
            # for bn in VARIABLES[variable]["bn"]:
                predicted = estimator.explain(explicand, actual_sample_size)
                filename = f"output/{estimator_class_name}_{dataset}_{weighting}.csv"
                if not os.path.exists(filename):
                    with open(filename, 'w') as f:
                        f.write('')

                with open(filename, 'a') as f:
                    dict = {
                        'sample_size': actual_sample_size,
                        'difference': noised_model.get_sample_count() - actual_sample_size,
                        'noise': noise_std,
                        'n' : n,
                        'error' : np.sum((predicted - true_values[idx]) ** 2) / np.sum(true_values[idx] ** 2),
                        'sum_error' : (np.sum(predicted) - np.sum(true_values[idx])) ** 2 / np.sum(true_values[idx])**2,
                        'sum_true' : np.sum(true_values[idx]),
                    }
                    f.write(str(dict) + '\n')
                noised_model.reset_sample_count()

def run_debug():
    dataset = load_dataset("NHANES")
    X, y = dataset
    n = X.shape[1]
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, max_features=n)
    model.fit(X, y)
    baseline, explicands = load_input(X, num_runs=1, is_synthetic=False)
 
    weighting = "shapley"
    weighting = "beta_shapley_16_16"
    #weighting = "banzhaf"

    estimator_names = [
        "RegressionAdjusted",
        "KernelBanzhaf+",
    ]

    estimators = {
        name: ESTIMATOR_FACTORY[name](model, baseline, weighting=weighting) for name in estimator_names
    }

    true_values = tree_prob(baseline, explicands, model, weighting)
    # print(f"True values: {true_values}")

    for sample_size in [8, 16, 32]:
        for idx, explicand in enumerate(explicands):
            results = {}
            for estimator_name, estimator in estimators.items():
                #if estimator_name == "UniversalConstrained":
                #    results[estimator_name] = estimator.explain(explicand, sample_size * n, true_sum=np.sum(true_values[idx]))
                #    #results[estimator_name] = estimator.explain(explicand, sample_size * n, true_sum=estimated_sum)
                #else:
                results[estimator_name] = estimator.explain(explicand, sample_size * n)
            
            for estimator_name, result in results.items():
                print(f"Estimator: {estimator_name} \t Error: {((result - true_values[idx]) ** 2).sum()/ (true_values[idx] ** 2).sum()}")
                #print(f"Estimate: {result}")
                #print(f"Sum difference: {np.sum(result) - np.sum(true_values[idx])}")

def check_fit():
    n = 10
    weighting = "shapley"
    p = get_p(n, weighting)
    model = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(50, 100, 50, 100, 50, 100, 50), max_iter=300)

    sample_multiplier = 10

    X, y = sklearn.datasets.make_regression(n_samples=100, n_features=n)
    model.fit(X, y)

    baseline, explicands = load_input(pd.DataFrame(X), num_runs=1, is_synthetic=False)
    idx = 0


    estimator = ESTIMATOR_FACTORY['GeneralMonteCarlo'](model, baseline, weighting=weighting, regression_adjust="tree")
    reg_model = estimator.explain(explicands[idx], n * sample_multiplier, return_direct=True)

    # shape: (2**n, 1)
    values = np.arange(2**n, dtype=np.uint32)[:, None]  
    # Create powers of 2 to mask bits
    # shape: (n,)
    powers = 2 ** np.arange(n - 1, -1, -1, dtype=np.uint32)
    # Apply bitmask and compare to 0 to get bits
    all_binary = ((values & powers) > 0).astype(int)

    full_input = baseline * (1-all_binary) + explicands[idx] * (all_binary)
    v = model.predict(full_input)
    v_pred = reg_model.predict(full_input)

    error = 0
    for i in range(n):
        with_i = all_binary[:, i] == 1
        without_i = ~with_i  # or: all_binary[:, i] == 0

        subset_sizes_with_i = all_binary[with_i].sum(axis=1) - 1
        subset_sizes_without_i = all_binary[without_i].sum(axis=1)

        error += (
            p[subset_sizes_with_i] * (v[with_i] - v_pred[with_i])**2
            + p[subset_sizes_without_i] * (v[without_i] - v_pred[without_i])**2
        ).sum()
    print(v[:10])
    print(v_pred[:10])
    print(p)
    print(error)
    print(((v-v_pred)**2).mean()/((v)**2).mean())

    
def parse_args():
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument(
       "--variable",
        type=str,
        default="basic",
        help="basic, sample_size, or noise"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--check_fit",
        action="store_true",
        help="Run experiments on quality of fit"
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
    variable = args.variable

    if args.debug:
        run_debug()
        exit()
    
    if args.check_fit:
        check_fit()
        exit()

    if not os.path.exists("output"):
        os.makedirs("output")
    experiments = generate_experiment_configs()
    print(experiments)
    datasets = DATASETS["small_n"] + DATASETS["big_n"]

    for dataset in datasets:
        X, y = load_dataset(dataset)
        n = X.shape[1]
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42, max_features=n)
        model.fit(X, y)
        baseline, explicands = load_input(X, NUM_RUNS, is_synthetic=False)
        for exp_cfg in experiments:
            logging.info(f"Running experiment: {exp_cfg['name']} on {dataset}")
            run_experiment(exp_cfg, baseline, explicands, model, variable, dataset)

if __name__ == "__main__":
    main()
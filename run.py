from typing import Any, Dict
import argparse
import logging
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import xgboost as xgb

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

def get_feature_data(rng, n, sample_multiplier, baseline, explicand, model):
    num_samples = min(n * sample_multiplier, 2**n)
    binary = rng.integers(0, 2, size=(num_samples, n), dtype=np.uint8)
    model_input = baseline * (1-binary) + explicand * (binary)
    label = model.predict(model_input)

    return label, model_input

def check_fit():
    rng = np.random.default_rng()
    num_runs = 10

    for dataset in DATASETS["small_n"] + DATASETS["big_n"]:
        filename = f"output/fit_{dataset}.csv"
        X, y = load_dataset(dataset)
        n = X.shape[1]

        ground_truth_classes = {
            'Linear' : sklearn.linear_model.LinearRegression(),
            'RandomForest' : sklearn.ensemble.RandomForestRegressor(),
            'NeuralNet' : sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(50, 100, 50, 100, 50, 100, 50), max_iter=300),
        }

        for ground_truth_class, model in ground_truth_classes.items():
            model.fit(X.values, y)
            baseline, explicands = load_input(pd.DataFrame(X), num_runs=num_runs, is_synthetic=False)

            for explicand in explicands:

                test_label, test_input = get_feature_data(rng, n, 32, baseline, explicand, model)

                sample_multipliers = [2, 4, 8, 16, 32, 64, 128]
                sample_multipliers = [64, 128]

                for sample_multiplier in sample_multipliers:
                    train_label, train_input = get_feature_data(rng, n, sample_multiplier, baseline, explicand, model)
                    
                    reg_model_classes = ["Linear", "RandomForest", "XGBoost"]

                    for reg_model_class in reg_model_classes:
                        if reg_model_class == "Linear":
                            reg_model = sklearn.linear_model.LinearRegression()
                        elif reg_model_class == "RandomForest":
                            reg_model = sklearn.ensemble.RandomForestRegressor()
                        elif reg_model_class == "XGBoost":
                            reg_model = xgb.XGBRegressor()
                        else:
                            raise ValueError("Invalid regression model class, must be one of: Linear, RandomForest, XGBoost")

                        reg_model.fit(train_input, train_label)
                        test_pred = reg_model.predict(test_input)
                        error = np.sum((test_pred - test_label) ** 2) / np.sum(test_label ** 2)

                        dict = {
                            'ground_truth_class' : ground_truth_class,
                            'reg_model_class' : reg_model_class,
                            'sample_multiplier' : sample_multiplier,
                            'error' : error,

                        }
                        with open(filename, 'a') as f:
                            f.write(str(dict) + '\n')
    
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
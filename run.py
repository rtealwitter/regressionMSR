from typing import Any, Dict
import argparse
import logging
import sklearn.ensemble
import sklearn.linear_model
import sklearn.neural_network
import xgboost as xgb
import time

from kernelprob.configs.config import generate_experiment_configs, ESTIMATOR_FACTORY, DATASETS, VARIABLES, NUM_RUNS
from kernelprob.utils.datasets import load_dataset, load_input
from kernelprob.utils.p_generator import get_p
from kernelprob.utils.run_utils import *
from kernelprob.exact.treeprob import tree_prob
from kernelprob.exact.treeshap import tree_shap
from kernelprob.exact.treeprob_v2 import tree_prob_v2
from kernelprob.exact.enumeration import enumeration_prob

def run_experiment(exp_cfg: Dict[str, Any], baseline, explicands, model: Any, variable: str, dataset: str, true_values: dict, ground_truth_class: str):
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
        return
    true_values = true_values[weighting]
    
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

    for noise_std in VARIABLES[variable]["noise"]:
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
            if actual_sample_size >= 2**n: 
                logging.warning(f"Sample size {actual_sample_size} exceeds 2^n for n={n}. Skipping.")
                continue
            if actual_sample_size >= 5e4:
                logging.warning(f"Sample size {actual_sample_size} exceeds 5e4. Skipping to avoid long computation time.")
                continue
            for idx, explicand in enumerate(explicands):
            # for bn in VARIABLES[variable]["bn"]:
                time_start = time.time()
                predicted = estimator.explain(explicand, actual_sample_size)
                filename = f"output/{estimator_class_name}_{dataset}_{weighting}.csv"
                if not os.path.exists(filename):
                    with open(filename, 'w') as f:
                        f.write('')

                with open(filename, 'a') as f:
                    dict = {
                        'ground_truth_class' : ground_truth_class,
                        'sample_size': actual_sample_size,
                        'difference': noised_model.get_sample_count() - actual_sample_size,
                        'noise': noise_std,
                        'n' : n,
                        'error' : np.sum((predicted - true_values[idx]) ** 2) / np.sum(true_values[idx] ** 2),
                        'sum_error' : (np.sum(predicted) - np.sum(true_values[idx])) ** 2 / np.sum(true_values[idx])**2,
                        'sum_true' : np.sum(true_values[idx]),
                        'time' : time.time() - time_start,
                    }
                    f.write(str(dict) + '\n')
                noised_model.reset_sample_count()

def run_case_study():
    weighting = 'shapley'
    sample_size_mult = 5
    estimator_names = [
        'WSL',
        'MSR',
#        'LeverageSHAP',
        'LinearMSR',
        'TreeMSR',
    ]
#    weighting = "beta_shapley_8_8"
#    estimator_names = [
#        'WSL', 'LinearMSR', 'TreeMSR'
#    ]
    for dataset in DATASETS['small_n'] + DATASETS['big_n']:
        X, y = load_dataset(dataset)
        n = X.shape[1]
        baseline, explicands = load_input(X, num_runs=1)
        results = {}

        print(f"Dataset: {dataset} (n={n})")
        if n >= 20:
            model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, max_features=n)
            model.fit(X.values, y)
            results['True'] = tree_prob(baseline, explicands, model, weighting)
        else:
            model = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(50, 100, 50, 100, 50, 100, 50), max_iter=300)
            model.fit(X.values, y)
            results['True'] = enumeration_prob(baseline, explicands, model, weighting)
    
        estimators = {
            name: ESTIMATOR_FACTORY[name](model, baseline, weighting=weighting) for name in estimator_names
        }

        for estimator_name, estimator in estimators.items():
            results[estimator_name] = estimator.explain(explicands[0], sample_size_mult * n)
        
        with open(f"output/case_study_{dataset}_{weighting}.csv", 'w') as f:
            f.write(str(results).replace('\n', ' '))

def run_debug():
    dataset_name = "Bike Sharing"
    X, y = load_dataset(dataset_name)
    n = X.shape[1]
    print(fr"Dataset: {dataset_name} (n={n})")
    model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, max_features=n)
    model.fit(X.values, y)
    baseline, explicands = load_input(X, num_runs=10)
 
    weighting = "banzhaf"
    #weighting = "beta_shapley_2_2"
    # weighting = "banzhaf"
    # weighting = "weighted_banzhaf_0.8"
    # weighting = "random_42"
    print(f"Weighting: {weighting}")

    true_values = tree_prob(baseline, explicands, model, weighting)
    # true_values_v2 = tree_prob_v2(baseline, explicands, model, weighting)
    # enumerate_values = enumeration_prob(baseline, explicands, model, weighting)

    # # print(f"True values: {true_values}")
    # print(f"True values v2: {true_values_v2}")
    # print(f"Enumerate values: {enumerate_values}\n")

    # for i in range(len(enumerate_values)):
    #     if not np.allclose(enumerate_values[i], true_values_v2[i]):
    #         print(f"True values differ at index {i}\n")

    # exit()
    
    estimator_names = [
        "LinearMSR",
        "TreeMSR",
        "LeverageSHAP",
        #"OFA_S",
        # "OFA_A",
        #"GELS",
        #"ARM",
        # "WSL",
        #"WeightedSHAP",
        #"AME",
        # "ImprovedAME",
        # "GELS_S",
    ]

    estimators = {
        name: ESTIMATOR_FACTORY[name](model, baseline, weighting=weighting) for name in estimator_names
    }

    # Enumeration
    #true_values = enumeration_prob(baseline, explicands, model, weighting)
    # print(f"True values: {true_values}")
    num_reps = 10

    def fancy_round(x, precision=2):
        return f"{x:.{precision}e}"

    print(f'{"Estimator":<20} {"Mean":<10} {"1st":<10} {"2nd":<10} {"3rd":<10} {"Max":<10}')
    for sample_size in [6*n, 10*n, 20*n]:
        print("-" * 70)
        for estimator_name, estimator in estimators.items():
            errors = []
            for idx, explicand in enumerate(explicands):                
                estimate = estimator.explain(explicand, sample_size * n)
                errors += [
                    np.sum((estimate - true_values[idx]) ** 2) / np.sum(true_values[idx] ** 2)
                ]
            print(f"{estimator_name:<20} {fancy_round(np.mean(errors)):<10} {fancy_round(np.partition(errors, 0)[0]):<10} "
                  f"{fancy_round(np.partition(errors, 1)[1]):<10} {fancy_round(np.partition(errors, 2)[2]):<10} "
                  f"{fancy_round(np.max(errors)):<10}")
        


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
            baseline, explicands = load_input(pd.DataFrame(X), num_runs=num_runs)

            for explicand in explicands:

                test_label, test_input = get_feature_data(rng, n, 32, baseline, explicand, model)

                sample_multipliers = [2, 4, 8, 16, 32, 64, 128]

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
    parser.add_argument(
        "--case_study",
        action="store_true",
        help="Run case study experiment"
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

def get_true_values(model, baseline, explicands, experiments, ground_truth_class):
    true_values = {}
    for exp_cfg in experiments:
        weighting = exp_cfg.get("weighting", None)
        if weighting is None:
            logging.error(f"Weighting not specified for '{exp_cfg['name']}'. Skipping.")
            continue
        if weighting not in true_values:
            if "random" in weighting or ground_truth_class == "MLP":
                logging.info("Using enumeration.")
                true_values[weighting] = enumeration_prob(baseline, explicands, model, weighting)
            # TODO: remove if treeprob is good enough :)
            elif weighting == "shapley":
                logging.info("Using tree SHAP.")
                true_values[weighting] = tree_shap(baseline, explicands, model)
            else:
                logging.info(f"Using tree prob for weighting: {weighting}")
                true_values[weighting] = tree_prob(baseline, explicands, model, weighting)
    return true_values

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
    
    if args.case_study:
        run_case_study()
        exit()

    if not os.path.exists("output"):
        os.makedirs("output")
    experiments = generate_experiment_configs()
    print(experiments)

    np.random.seed(42)
    ground_truth_class = 'MLP'
    for dataset in DATASETS["small_n"]:
        X, y = load_dataset(dataset)
        n = X.shape[1]
        model = sklearn.neural_network.MLPRegressor(hidden_layer_sizes=(50, 100, 50, 100, 50, 100, 50), max_iter=300)
        model.fit(X.values, y)
        baseline, explicands = load_input(X, NUM_RUNS)
        true_values = get_true_values(model, baseline, explicands, experiments, ground_truth_class)
        for exp_cfg in experiments:
            logging.info(f"Running experiment: {exp_cfg['name']} on {dataset}")
            run_experiment(exp_cfg, baseline, explicands, model, variable, dataset, true_values=true_values, ground_truth_class=ground_truth_class)

    ground_truth_class = 'RandomForest'
    for dataset in DATASETS["big_n"]:
        X, y = load_dataset(dataset)
        n = X.shape[1]
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42, max_features=n)
        model.fit(X.values, y)
        baseline, explicands = load_input(X, NUM_RUNS)
        true_values = get_true_values(model, baseline, explicands, experiments, ground_truth_class)
 
        for exp_cfg in experiments:
            logging.info(f"Running experiment: {exp_cfg['name']} on {dataset}")
            run_experiment(exp_cfg, baseline, explicands, model, variable, dataset, true_values=true_values, ground_truth_class=ground_truth_class)

if __name__ == "__main__":
    main()
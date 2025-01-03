import matplotlib.pyplot as plt
from .estimators import *
from .datasets import *
import numpy as np
import xgboost as xgb
import os
from tqdm import tqdm
import scipy

# Every line of output files contains a dictionary with the following keys
# 'sample_size': number of samples used to estimate SHAP values
# 'noise': standard deviation of noise added to the labels
# 'shap_error': mean squared error between estimated and true SHAP values
# 'weighted_error' (optional): ||Ax- b||^2 / ||Ax* - b||^2 where x* is the true SHAP values and x is the estimated SHAP values
# 'gamma' (optional): ||b||^2 / ||Ax||^2 where x is the estimated SHAP values

def build_full_linear_system(baseline, explicand, model):
    n = baseline.shape[1]
    binary_Z = np.zeros((2**n-2, n))
    idx = 0
    for s in range(1, n):
        for indices in itertools.combinations(range(n), s):
            binary_Z[idx, list(indices)] = 1
            idx += 1
    binary_Z1_norm = np.sum(binary_Z, axis=1)
    inv_sqrt_weights = np.sqrt(binary_Z1_norm * (n - binary_Z1_norm) * scipy.special.binom(n, binary_Z1_norm))

    Z = 1 / inv_sqrt_weights[:, np.newaxis] * binary_Z
    P = np.eye(n) - np.ones((n, n)) / n
    A = Z @ P
    inputs = baseline * (1 - binary_Z) + explicand * binary_Z
    v1 = model.predict(explicand)
    vz = model.predict(inputs)
    v0 = model.predict(baseline)
    y = (vz - v0) / inv_sqrt_weights
    b = y - Z.sum(axis=1) * (v1 - v0) / n
    return {'A': A, 'b': b}

def get_dataset_size(dataset):
    if 'Synthetic_' in dataset:
        return int(dataset.split('_')[1])
    X, y = load_dataset(dataset)
    return X.shape[1]

def read_file(dataset, estimator, x_name, y_name, constraints={}):
    filename = f'output/{dataset}_{estimator}.csv'
    if not os.path.exists(filename): return {}
    results = {}
    with open(filename, 'r') as f:
        for line in f:
            dict = eval(line)
            add = True
            for key, value in constraints.items():
                if dict[key] != value:
                    add = False
            if add:
                try:
                    x, y = dict[x_name], dict[y_name]
                    if x not in results:
                        results[x] = []
                    results[x].append(y)
                except KeyError:
                    pass
    return results

def load_results(datasets, x_name, y_name, constraints, estimator_names=estimators.keys(), is_actual_sample_size=False):
    results_by_dataset = {}
    original_sample_size = constraints.get('sample_size', 1)
    for dataset in datasets:
        n = get_dataset_size(dataset)
        if 'sample_size' in constraints and not is_actual_sample_size:
            constraints['sample_size'] = int(original_sample_size * n)
        results_by_estimator = {}
        for estimator_name in estimator_names:
            if estimator_name == 'Official Tree SHAP':
                continue
            results = read_file(dataset, estimator_name, x_name, y_name, constraints)
            if results != {}:
                results_by_estimator[estimator_name] = results
        if results_by_estimator != {}:
            results_by_dataset[dataset] = results_by_estimator
    return results_by_dataset

def compute_weighted_error(baseline, explicand, model, shap_values):
    n = baseline.shape[1]
    Z = np.zeros((2**n-2, n))
    idx = 0
    for s in range(1, n):
        for indices in itertools.combinations(range(n), s):
            Z[idx, list(indices)] = 1
            idx += 1
    Z1_norm = np.sum(Z, axis=1)
    inv_weights = Z1_norm * (n - Z1_norm) * scipy.special.binom(n, Z1_norm)
    weights = 1 / inv_weights
    inputs = baseline * (1 - Z) + explicand * Z
    vz = model.predict(inputs)
    v0 = model.predict(baseline)
    return np.sum(weights * (shap_values @ Z.T - (vz - v0)) ** 2)

markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'P', '*', 'X', 'd', 'h', 'H', '+', 'x', '|', '_']

def visualize_predictions(datasets, include_estimators, filename):
    plt.clf()
    row_num = 2 if len(include_estimators) > 3 else 1
    fig, axs = plt.subplots(row_num, 3, figsize=(10, 3 * row_num))
    for dataset_idx, dataset in enumerate(datasets):
        X, y = load_dataset(dataset)
        n = X.shape[1]
        num_samples = 5 * n
        model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
        model.fit(X, y)
        baseline, explicand = load_input(X)
        # 2 by 3 array of axes in matplotlib plot
        true_shap_values = estimators['Official Tree SHAP'](baseline, explicand, model, num_samples).flatten()
        # Ensure magnitude of true SHAP values is at most 1
        normalizing_scale = np.max(np.abs(true_shap_values))
        true_shap_values /= normalizing_scale
        i = 0
        for estimator_name, estimator in estimators.items():
            if estimator_name not in include_estimators:
                continue
            shap_values = estimator(baseline, explicand, model, num_samples).flatten()
            # Ensure magnitude of estimated SHAP values is at most 1
            shap_values /= normalizing_scale
            if row_num == 1:
                ax = axs[i]
            else:
                ax = axs[i // 3, i % 3]
            ax.scatter(true_shap_values, shap_values, alpha=0.5, marker=markers[dataset_idx], label=dataset + ' (n=' + str(n) + ')')
            ax.set_title(estimator_name)
            i += 1
    
    for ax in axs.flatten():
        # Plot the line y = x
        ax.plot([ax.get_xlim()[0], ax.get_xlim()[1]], [ax.get_xlim()[0], ax.get_xlim()[1]], color='gray', alpha=0.5)

    # Set x label for bottom row
    if row_num == 2:
        for ax in axs[1]:
            ax.set_xlabel(r'True Shapley Values ($\phi$)')
        # Set y label for left column
        for ax in axs[:,0]:
            ax.set_ylabel(r'Predicted Shapley Values ($\tilde{\phi}$)')     
    else:
        for i, ax in enumerate(axs):
            if i == 0:
                ax.set_ylabel(r'Predicted Shapley Values ($\tilde{\phi}$)')
            ax.set_xlabel(r'True Shapley Values ($\phi$)')
            

    plt.legend(fancybox=True, bbox_to_anchor=(1,-.3), ncol=4)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.clf()

class NoisyModel:
    def __init__(self, model, noise_std):
        self.model = model
        self.noise_std = noise_std
        self.sample_count = 0

    def predict(self, X):
        self.sample_count += len(X)
        return self.model.predict(X) + np.random.normal(0, self.noise_std, X.shape[0])
    
    def get_sample_count(self):
        return self.sample_count

def run_small_setup(baseline, explicand, model, true_shap_values):
    linear_system = build_full_linear_system(baseline, explicand, model)
    best_weighted_error = np.sum((linear_system['A'] @ true_shap_values - linear_system['b'])**2)
    Aphi = linear_system['A'] @ true_shap_values
    gamma = np.sum((Aphi - linear_system['b'])**2) / np.sum((Aphi)**2)    
    normalized_gamma = gamma / np.sum((true_shap_values)**2)
    # Round to 2 significant figures
    normalized_gamma = float(f'{normalized_gamma:.2g}')
    return {'A': linear_system['A'], 'b': linear_system['b'], 'best_weighted_error': best_weighted_error, 'normalized_gamma': normalized_gamma, 'gamma': gamma}

def run_one_iteration(X, seed, dataset, model, sample_size, noise_std, num_runs, current_estimators):
    baseline, explicand = load_input(X, seed=seed, is_synthetic=dataset=='Synthetic')
    n = X.shape[1]
    is_small = 2**n <= 1e7
    # Compute the true SHAP values (assuming tree model)
    true_shap_values = estimators['Official Tree SHAP'](baseline, explicand, model, sample_size).flatten()

    small_setup = {}
     
    for estimator_name, estimator in current_estimators.items():        
        if estimator_name in ['Official Tree SHAP']:
            continue

        results = read_file(dataset, estimator_name, 'sample_size', 'shap_error', {'noise': noise_std, 'n': n})
        if results != {} and sample_size in results:
            if len(results[sample_size]) >= num_runs: continue
        noised_model = NoisyModel(model, noise_std)
        shap_values = estimator(baseline, explicand, noised_model, sample_size).flatten()

        filename = f'output/{dataset}_{estimator_name}.csv'
        if not os.path.exists(filename):
            with open(filename, 'w') as f:
                f.write('')

        with open(filename, 'a') as f:
            dict = {
                'sample_size': sample_size,
                'difference': noised_model.get_sample_count() - sample_size,
                'noise': noise_std,
                'n' : n,
            }
            dict['shap_error'] = ((shap_values - true_shap_values) ** 2).mean()
            if is_small:
                if small_setup == {}:
                    small_setup = run_small_setup(baseline, explicand, model, true_shap_values)
                weighted_error = np.sum((small_setup['A'] @ shap_values - small_setup['b'])**2)
                dict['weighted_error'] = weighted_error / small_setup['best_weighted_error'] 
            f.write(str(dict) + '\n')

def compute_gamma(dataset, seed=42):
    X, y = load_dataset(dataset)
    n = X.shape[1]
    is_small = 2**n <= 1e7
    if not is_small: return {}
    # Assuming deterministic
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model.fit(X, y)
    baseline, explicand = load_input(X, seed=seed, is_synthetic=dataset=='Synthetic')
    true_shap_values = estimators['Official Tree SHAP'](baseline, explicand, model, num_samples=0).flatten()
    small_setup = run_small_setup(baseline, explicand, model, true_shap_values)
    return {
        'gamma': small_setup['gamma'],
        'normalized_gamma' : small_setup['normalized_gamma']
    }


def benchmark(num_runs, dataset, current_estimators, hyperparameter, hyperparameter_values, silent=False):              

    X, y = load_dataset(dataset)
    n = X.shape[1]
    # Assuming deterministic
    model = xgb.XGBRegressor(n_estimators=100, max_depth=4)
    model.fit(X, y)

    config = {'sample_size': 10*n, 'noise_std' : 0}
    for run_idx in tqdm(range(num_runs), disable=silent):
        for hyperparameter_value in hyperparameter_values:
            if hyperparameter == 'sample_size':
                hyperparameter_value = int(hyperparameter_value * n)
            config[hyperparameter] = hyperparameter_value
            run_one_iteration(X, run_idx * num_runs, dataset, model, sample_size=config['sample_size'], noise_std=config['noise_std'], num_runs=num_runs, current_estimators=current_estimators)

class SyntheticModel:
    def __init__(self, v, correspondence):
        self.v = v
        self.num_samples = 0
        self.correspondence = correspondence

    def predict(self, X):
        # X is a binary matrix
        # Get the integer represented in each row
        indices = np.sum(2**np.arange(X.shape[1]) * X, axis=1).astype(int)
        # Get the corresponding index in v
        self.num_samples += len(X)
        return self.v[[self.correspondence[i] for i in indices]]
    
    def get_sample_count(self):
        return self.num_samples

def build_gamma_labels(n, alpha):
    # Construct A
    binary_Z = np.zeros((2**n-2, n))
    idx = 0
    for s in range(1, n):
        for indices in itertools.combinations(range(n), s):
            binary_Z[idx, list(indices)] = 1
            idx += 1
    # Convert all rows to their integer form
    X = binary_Z
    indices = np.sum(2**np.arange(X.shape[1]) * X, axis=1).astype(int)
    correspondence = {0:0, 2**n-1:-1}
    for i in range(2**n-2):
        correspondence[indices[i]] = i + 1

    binary_Z1_norm = np.sum(binary_Z, axis=1)
    inv_sqrt_weights = np.sqrt(binary_Z1_norm * (n - binary_Z1_norm) * scipy.special.binom(n, binary_Z1_norm))
    Z = 1 / inv_sqrt_weights[:, np.newaxis] * binary_Z
    P = np.eye(n) - np.ones((n, n)) / n
    A = Z @ P

    # Perform QR decomposition of A
    Q, R = np.linalg.qr(A)

    # The last column of Q is orthogonal to all the columns of A
    # if A has full rank and is not square
    col_not_in_span = Q[:, -1]
    col_not_in_span = col_not_in_span / np.linalg.norm(col_not_in_span)
    
    # Check that r is orthogonal to the columns of A
    assert np.allclose(A.T @ col_not_in_span, 0)

    # Construct b as (1-alpha) * a column in span of A + alpha * a column not in span of A
#    xstar = np.random.randn(n)
#    col_in_span = A @ xstar
    col_in_span = A[:, 0]
    col_in_span = col_in_span / np.linalg.norm(col_in_span)
    b = (1 - alpha) * col_in_span + alpha * col_not_in_span

    # Convert from b to y
    v1 = 1
    v0 = 0
    y = b + Z.sum(axis=1) * (v1 - v0) /n 

    v = np.zeros(2**n)
    v[1:-1] = y * inv_sqrt_weights
    v[0] = v0
    v[-1] = v1

    # True SHAP values.
    # Solve Ax = b
    true_shap_values = np.linalg.lstsq(A, b, rcond=None)[0]
    best_weighted_error = np.sum((A @ true_shap_values - b)**2)

    gamma = np.sum((A @ true_shap_values - b)**2) / np.sum((A @ true_shap_values)**2)

    return {'v': v, 'true_shap_values': true_shap_values, 'best_weighted_error': best_weighted_error, 'correspondence': correspondence}

def benchmark_gamma(num_runs, n, include_estimators, sample_size, silent=False):
    baseline = np.zeros((1, n))
    explicand = np.ones((1, n))
    for run_idx in tqdm(range(num_runs), disable=silent):
        for alpha in [.2, .3, .4, .5, .6, .7, .8]:
            seed = run_idx * num_runs + int(alpha * 100)
            np.random.seed(seed)
            gamma_labels = build_gamma_labels(n, alpha)

            is_small = 2**n <= 1e7

            small_setup = {}

            dataset = 'Synthetic_' + str(n)
            
            for estimator_name, estimator in estimators.items():
                if estimator_name not in include_estimators:
                    continue
                model = SyntheticModel(gamma_labels['v'], gamma_labels['correspondence'])
                results = read_file(dataset, estimator_name, 'alpha', 'shap_error', {'n': n})
                if results != {} and alpha in results:
                    if len(results[alpha]) >= num_runs: continue
                shap_values = estimator(baseline, explicand, model, sample_size).flatten()

                filename = f'output/{dataset}_{estimator_name}.csv'

                with open(filename, 'a') as f:
                    dict = {
                        'sample_size': sample_size,
                        'difference': model.get_sample_count() - sample_size,
                        'noise': 0,
                        'n' : n,
                        'alpha' : alpha,
                    }
                    dict['shap_error'] = ((shap_values - gamma_labels['true_shap_values']) ** 2).mean()
                    if is_small:
                        if small_setup == {}:
                            small_setup = run_small_setup(baseline, explicand, model, gamma_labels['true_shap_values'])
                        weighted_error = np.sum((small_setup['A'] @ shap_values - small_setup['b'])**2)
                        dict['weighted_error'] = weighted_error / gamma_labels['best_weighted_error'] 
                        dict['gamma'] = small_setup['gamma']
                    f.write(str(dict) + '\n')
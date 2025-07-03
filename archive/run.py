import regressionMSR as rmsr
import numpy as np

small_n = ['IRIS', 'California', 'Diabetes', 'Adult']

small_n = ['Adult']

big_n = ['Correlated', 'Independent', 'NHANES', 'Communities']

def get_hyperparameter_values(name):
    if name == 'noise_std': return [0]
    if name == 'noise_std':
        return [0, .5 * 1e-3, 1e-3, .5 * 1e-2, 1e-2, .5 * 1e-1, 1e-1, .5, 1]
    elif name == 'sample_size':
        return [5, 10, 20, 40, 80, 160]
    else:
        raise ValueError(f'Unknown hyperparameter {name}')

# Debugging access to official SHAP estimators
#import logging
#log = logging.getLogger('shap')
#logging.basicConfig(level=logging.DEBUG)

datasets = small_n + big_n

if False:
    gammas = {x : [] for x in small_n}
    for seed in range(100):
        for dataset in small_n:
            gamma_run = rmsr.compute_gamma(dataset, seed=seed)
            gammas[dataset].append(gamma_run['gamma'])

        if seed % 10 != 0: continue
        print()
        for dataset in small_n:
            # print dataset name, 1st quartile, median, 3rd quartile
            print(dataset, np.percentile(gammas[dataset], 25), np.median(gammas[dataset]), np.percentile(gammas[dataset], 75))
            

#rmsr.plot_probs([10,100,1000], folder='images/')

if False:

    rmsr.visualize_predictions(datasets, main_estimators, filename='images/main_detailed.pdf')
    #rmsr.visualize_predictions(datasets, ablation_estimators, filename='images/ablation_detailed.pdf')

if True:
    # weighting = 'shapley'
    # estimators = {
    #     'Kernel Prob': rmsr.estimators['Kernel Prob'],
    #     'Leverage SHAP': rmsr.estimators['Leverage SHAP'],
    # }
    estimators = {
        'Constrained Regression': rmsr.estimators['Constrained Regression'],
        'Kernel Banzhaf': rmsr.estimators['Kernel Banzhaf'],
        'Unconstrained Regression': rmsr.estimators['Unconstrained Regression'],
        'Leverage SHAP' : rmsr.estimators['Leverage SHAP'],
    }
    weighting = 'shapley'
    
    num_runs = 5
    for dataset in small_n + big_n:
        print(dataset)
        for hyperparameter in ['sample_size']:#, 'noise_std']:
            print(hyperparameter)
            rmsr.benchmark(num_runs, dataset, estimators, hyperparameter, get_hyperparameter_values(hyperparameter), weighting=weighting)

## Plots
#
#for y_name in ['shap_error', 'weighted_error']:
#    # Performance by number of samples
#    x_name = 'sample_size'
#    constraints = {'noise': 0}
#    results = rmsr.load_results(small_n + big_n, x_name, y_name, constraints)
#    rmsr.plot_with_subplots(results, x_name, y_name, filename=f'images/main_{x_name}-{y_name}.pdf', log_x=True, log_y=y_name == 'shap_error', include_estimators=main_estimators, plot_mean=False)
#    rmsr.plot_with_subplots(results, x_name, y_name, filename=f'images/ablation_{x_name}-{y_name}.pdf', log_x=True, log_y=y_name == 'shap_error', include_estimators=ablation_estimators, plot_mean=False)
#
#    # Performance by noise level
#    x_name = 'noise'
#    constraints = {'sample_size': 10}
#    results = rmsr.load_results(small_n + big_n, x_name, y_name, constraints)
#    rmsr.plot_with_subplots(results, x_name, y_name, filename=f'images/main_{x_name}-{y_name}.pdf', log_x=True, log_y=y_name == 'shap_error', include_estimators=main_estimators, plot_mean=False)
#    rmsr.plot_with_subplots(results, x_name, y_name, filename=f'images/ablation_{x_name}-{y_name}.pdf', log_x=True, log_y=y_name == 'shap_error', include_estimators=ablation_estimators, plot_mean=False)
#
## Tables
#for y_name in ['shap_error', 'weighted_error']:
#    results = rmsr.load_results(small_n + big_n, 'sample_size', y_name, {'noise': 0, 'sample_size' : 10})
#    rmsr.one_big_table(results, f'tables/ablation_{y_name}.tex', error_type=y_name)
#    results_main = {}
#    for dataset in results:
#        results_main[dataset] = {estimator : results[dataset][estimator] for estimator in main_estimators}
#    rmsr.one_big_table(results_main, f'tables/main_{y_name}.tex', error_type=y_name)
#        
#    #for dataset in results:
#    #    rmsr.benchmark_table(results[dataset], f'tables/{dataset}-{y_name}.tex', print_md=False)
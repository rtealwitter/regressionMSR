import kernelprob as kp

ablation_estimators = ['Kernel SHAP', 'Optimized Kernel SHAP', 'Leverage SHAP', 'Kernel SHAP Paired', 'Leverage SHAP wo Bernoulli', 'Leverage SHAP wo Bernoulli, Paired']

main_estimators = ['Kernel SHAP', 'Optimized Kernel SHAP', 'Leverage SHAP']

ns = [10, 12, 14, 16]
sample_size = 500
num_runs = 100
if False:
    for n in ns:
        kp.benchmark_gamma(num_runs, n, ablation_estimators, sample_size=sample_size, silent=False)

# Plots

for y_name in ['shap_error', 'weighted_error']:
    # Performance by gamma
    x_name = 'gamma'
    constraints = {'sample_size': sample_size, 'noise': 0}
    results = kp.load_results([f'Synthetic_{n}' for n in ns], x_name, y_name, constraints, is_actual_sample_size=True)
    kp.plot_with_subplots(results, x_name, y_name, filename=f'images/main_gamma_{y_name}.pdf', log_x=True, log_y=False, include_estimators=main_estimators, plot_mean=False)
    kp.plot_with_subplots(results, x_name, y_name, filename=f'images/ablation_gamma_{y_name}.pdf', log_x=True, log_y=False, include_estimators=ablation_estimators, plot_mean=True)
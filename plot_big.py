from kernelprob.configs.config import *
import matplotlib.pyplot as plt

# Load in files into a giant dictionary
# filename format: dataset_estimator_weighting

dataset_names = DATASETS['small_n'] + DATASETS['big_n']

weightings = {
    'shapley' : ['shapley'],
    'banzhaf' : ['banzhaf'],
    'beta_shapley_sym' : ['beta_shapley_1_1', 'beta_shapley_2_2', 'beta_shapley_4_4', 'beta_shapley_8_8', 'beta_shapley_16_16'],
    'beta_shapley_asym' : ['beta_shapley_1_1', 'beta_shapley_1_2', 'beta_shapley_1_4', 'beta_shapley_1_8', 'beta_shapley_1_16'],
    'weighted_banzhaf' : ['weighted_banzhaf_0.5', 'weighted_banzhaf_0.6', 'weighted_banzhaf_0.7', 'weighted_banzhaf_0.8', 'weighted_banzhaf_0.9']
}

# General estimators in ESTIMATOR_ALL
# Shapley estiamtors in ESTIMATOR_SHAPLEY
# Banzhaf estimators in ESTIMATOR_BANZHAF

# Load in all the files into a dictionary called results


# Get list of files in outputs folder
import os


def extract_helper(filename, valid_names):
    for name in valid_names:
        if name in filename:
            return name
    raise ValueError(f'Could not find valid name in filename {filename}')

def load_filtered_error(filter):
    results = {}
    filenames = os.listdir('output')
    for filename in filenames:
        # The challenge is that there are extra underscores in the filename that we need to account for
        dataset = extract_helper(filename, dataset_names)
        estimator = extract_helper(filename, ESTIMATOR_FACTORY.keys())
        weighting = extract_helper(filename, weightings.keys())

        # Load in the file
        with open(f'output/{filename}', 'r') as f:
            # Each line is a dictionary, we want to e
            for line in f:
                line = eval(line)
                # Check if the line is valid
                if 'sample_size' in filter:
                    if line['sample_size'] // line['n'] not in filter['sample_size']:
                        continue
                if 'noise' in filter:
                    if line['noise'] not in filter['noise']:
                        continue
                # Add to the results
                if dataset not in results:
                    results[dataset] = {}
                if weighting not in results[dataset]:
                    results[dataset][weighting] = {}
                if estimator not in results[dataset][weighting]:
                    results[dataset][weighting][estimator] = []
                results[dataset][weighting][estimator].append(line['error'])
    return results


filter = {
    'sample_size' : [10],
    'noise' : [0],
}
results = load_filtered_error(filter)
print(results['Adult']['shapley'])                

            



    

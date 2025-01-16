import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import os

from .datasets import load_dataset, load_input
from ..configs.config import ESTIMATOR_FACTORY

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
    
    def reset_sample_count(self):
        self.sample_count = 0


def get_filename(dataset, estimator, weighting):
    return f'output/{weighting}_{dataset}_{estimator}.csv'

def read_file(dataset, estimator, x_name, y_name, weighting, constraints={}):
    filename = get_filename(dataset, estimator, weighting)
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


def get_dataset_size(dataset):
    if 'Synthetic_' in dataset:
        return int(dataset.split('_')[1])
    X, y = load_dataset(dataset)
    return X.shape[1]
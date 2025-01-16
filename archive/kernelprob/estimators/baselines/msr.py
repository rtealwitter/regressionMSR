import numpy as np
import warnings

# Only supposrts Banzhaf for now
def msr(baseline, explicand, model, num_samples, weighting="banzhaf"):
    warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
    n = baseline.shape[1]
    phi = np.zeros_like(baseline)
    gen = np.random.Generator(np.random.PCG64())

    subset_masks = gen.integers(0, 2, size=(num_samples, n))
    model_input = np.zeros((num_samples, n))
    model_input = baseline * (1-subset_masks) + explicand * subset_masks
    model_output = model.predict(model_input)

    for i in range(n):
        with_i = np.where(subset_masks[:, i] == 1)
        without_i = np.where(subset_masks[:, i] == 0)
        if with_i[0].shape[0] == 0 or without_i[0].shape[0] == 0:
            phi[:, i] = 0
            print(f"Feature {i} is not in any subset")
        else:
            phi[:, i] = np.mean(model_output[with_i]) - np.mean(model_output[without_i])

    return phi
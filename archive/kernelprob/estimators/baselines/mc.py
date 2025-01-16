import numpy as np
import warnings

def mc_shapley_sampler(i, except_i, gen):
    size = gen.choice(len(except_i))
    indices = gen.choice(except_i, size, replace=False)
    indices_with_i = np.append(i, indices)
    return indices, indices_with_i

def mc_banzhaf_sampler(i, except_i, gen):
    subset_mask = gen.integers(0, 2, size=len(except_i))
    indices = except_i[subset_mask == 1]
    indices_with_i = np.append(i, indices)
    return indices, indices_with_i

mc_sampler = {
    "shapley": mc_shapley_sampler,
    "banzhaf": mc_banzhaf_sampler
}

def monte_carlo(baseline, explicand, model, num_samples, weighting="shapley"):
    warnings.filterwarnings("ignore", message=".*does not have valid feature names.*")
    n = baseline.shape[1]
    samples_per_group = num_samples // n
    phi = np.zeros_like(baseline)
    gen = np.random.Generator(np.random.PCG64())
    for i in range(n):
        except_i = np.delete(range(n), i)    
        model_input = np.zeros((samples_per_group * 2, n))    
        sign = np.ones(samples_per_group * 2)
        for S_idx in range(samples_per_group):
            indices, indices_with_i = mc_sampler[weighting](i, except_i, gen)
            model_input[2*S_idx, indices] = 1
            model_input[2*S_idx + 1, indices_with_i] = 1
            sign[2*S_idx] = -1
        model_input = baseline * (1-model_input) + explicand * model_input
        model_output = model.predict(model_input)

        phi[:, i] = np.sum(model_output * sign) / samples_per_group

    return phi
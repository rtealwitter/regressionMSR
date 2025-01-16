import numpy as np
from sklearn.linear_model import LinearRegression


def kernel_banzhaf(baseline, explicand, model, num_samples, weighting="banzhaf"):
    n = baseline.shape[1]
    weight = 1 / 2
    phi = np.zeros_like(baseline)
    gen = np.random.Generator(np.random.PCG64())

    masks_original = gen.integers(0, 2, size=(num_samples // 2, n))
    masks_complement = 1 - masks_original
    masks = np.vstack((masks_original, masks_complement))

    model_input = baseline * (1 - masks) + explicand * masks

    lr_features = masks - weight
    lr_output = model.predict(model_input)

    l_model = LinearRegression().fit(lr_features, lr_output)
    values = l_model.coef_.T

    for i in range(n):
        phi[:, i] = values[i].tolist()
    
    return phi
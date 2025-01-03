import numpy as np

def monte_carlo(baseline, explicand, model, num_samples):
    n = baseline.shape[1]
    # Ensure that the number of samples is even
    samples_per_group = 2 * ((num_samples // n) // 2 )
    samples_per_group = samples_per_group if samples_per_group % 2 == 0 else samples_per_group + 1
    samples_per_group = num_samples
    phi = np.zeros_like(baseline)
    gen = np.random.Generator(np.random.PCG64())

    for i in range(n):
        except_i = np.delete(range(n), i)    
        model_input = np.zeros((samples_per_group, n))    
        sign = np.ones(samples_per_group)
        for S_idx in range(samples_per_group // 2):
            size = gen.choice(n)
            indices = gen.choice(except_i, size, replace=False)
            indices_with_i = np.append(i, indices)
            model_input[2*S_idx, indices] = 1
            model_input[2*S_idx + 1, indices_with_i] = 1
            sign[2*S_idx] = -1
        model_input = baseline * (1-model_input) + explicand * model_input
        model_output = model.predict(model_input)
        phi[:, i] = np.mean(model_output * sign)

    return phi





        

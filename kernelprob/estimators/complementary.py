import numpy as np
# Algorithm 2 in
# Efficient Sampling Approaches to Shapley Value Approximation by
# Jiayao Zhang, Qiheng Sun, Jinfei Liu, Li Xiong, Jian Pei, and Kui Ren

def complementary_contribution(baseline, explicand, model, num_samples):
    
    num_features = baseline.shape[1]
    gen = np.random.Generator(np.random.PCG64())
    eval_model = lambda X : model.predict(X)
    SV_ji = np.zeros((num_features, num_features))
    m_ji = np.zeros((num_features, num_features))
    for k in range(num_samples):
        pi_k = gen.permutation(num_features)
        i = gen.integers(0, num_features)
        S = pi_k[:i]
        N_S = pi_k[i:]
        S_input = np.copy(baseline)
        S_input[0, S] = explicand[0, S]
        N_S_input = np.copy(baseline)
        N_S_input[0, N_S] = explicand[0, N_S]
        u = eval_model(S_input) - eval_model(N_S_input)
        for j in range(i):
            SV_ji[pi_k[j], pi_k[i]] += u
            m_ji += 1
        for j in range(i, num_features):
            SV_ji[pi_k[j], pi_k[i]] -= u
            m_ji += 1
    phi = np.zeros((1,num_features))
    for i in range(num_features):
        SV_i = 0
        for j in range(num_features):
            SV_i += SV_ji[j, i] / m_ji[j, i]
        phi[0,i] = SV_i

    return phi
import numpy as np
import scipy
import scipy.special

def ith_combination(pool, r, index):
    # Function written by ChatGPT
    """
    Compute the index-th combination (0-based) in lexicographic order
    without generating all previous combinations.
    """
    n = len(pool)
    combination = []
    elements_left = n
    k = r
    start = 0
    
    for i in range(r):
        # Find the largest value for the first element in the combination
        # that allows completing the remaining k-1 elements
        for j in range(start, elements_left):
            count = scipy.special.binom(elements_left - j - 1, k - 1)
            if index < count:
                combination.append(pool[j])
                k -= 1
                start = j + 1
                break
            index -= count
    
    return tuple(combination)

def combination_generator(gen, n, s, num_samples):
    """
    Generate num_samples random combinations of s elements from a pool num_samples of size n in two settings:
    1. If the number of combinations is small (converting to an int does NOT cause an overflow error), randomly sample num_samples integers without replacement and generate the corresponding combinations on the fly with ith_combination.
    2. If the number of combinations is large (converting to an int DOES cause an overflow error), randomly sample num_samples combinations directly with replacement.
    """
    num_combos = int(scipy.special.binom(n, s))
    try:
        indices = gen.choice(num_combos, num_samples, replace=False)
        for i in indices:
            yield ith_combination(range(n), s, i)
    except OverflowError:
        for _ in range(num_samples):
            yield gen.choice(n, s, replace=False)

class ProbRegressionEstimator:
    def __init__(self, model, baseline, explicand, num_samples, p, paired_sampling=False, leverage_sampling=False, bernoulli_sampling=False):
        self.model = model
        self.baseline = baseline
        self.explicand = explicand
        # Ensure num_samples is even
        self.num_samples = int((num_samples) // 2) * 2
        self.paired_sampling = paired_sampling
        self.n = self.baseline.shape[1] # Number of features
        self.p = p
        self.gen = np.random.Generator(np.random.PCG64())

        # Compute n and p dependent constants
        sum1, sum2 = 0, 0
        for ell in range(self.n):
            sum1 += scipy.special.binom(self.n-1, ell) * p[ell]**2
            if ell >= 1:
                sum2 += scipy.special.binom(self.n-2, ell-1) * (p[ell]-p[ell-1])**2
        self.an = 2*sum1 - sum2
        self.bn = sum2 / self.an

        def leverage_score(s):
            first_term = 0 # |S| p_{|S|-1}^2 + (n-|S|) p_{|S|}^2
            second_term = 0 # (|S| p_{|S|-1} - (n-|S|) p_{|S|})^2
            if s > 0:
                first_term += s * self.p[s-1]**2
                second_term += s * self.p[s-1]
            if s < self.n:
                first_term += (self.n - s) * self.p[s]**2
                second_term -= (self.n - s) * self.p[s]
            # (first_term - bn / (1+n * bn) * second_term^2) / a_n
            return (first_term - self.bn / (1+self.n * self.bn) * second_term**2) / self.an
        
        self.sample_weight = lambda s : leverage_score(s) * scipy.special.binom(self.n, s)
        self.reweight = lambda s : 1 / self.sample_weight(s)
        self.kernel_weights = []
        self.sample = self.sample_with_replacement if not bernoulli_sampling else self.sample_without_replacement
        #self.used_indices = set()
    
    def add_helper(self, idx, indices, weight):
        size = len(indices)
        indices_complement = np.array([i for i in range(self.n) if i not in indices])
        if size > 0:
            self.A_tilde[idx, indices] = self.p[size-1]
        if size < self.n:
            self.A_tilde[idx, indices_complement] = -self.p[size]
        self.kernel_weights.append(weight)
    
    def add_one_sample(self, idx, indices, weight):
        if not self.paired_sampling:
            self.add_helper(idx, indices, weight)
        else:
            self.add_helper(2*idx, indices, weight)
            indices_complement = np.array([i for i in range(self.n) if i not in indices])
            self.add_helper(2*idx+1, indices_complement, weight)
    
    def sample_with_replacement(self):
        # NOT TESTED AFTER PORT TO PROBALISTIC VALUES
        self.A_tilde = np.zeros((self.num_samples, self.n))
        valid_sizes = np.array(list(range(self.n+1)))
        prob_sizes = self.sample_weight(valid_sizes)
        prob_sizes = prob_sizes / np.sum(prob_sizes)
        num_sizes = self.num_samples if not self.paired_sampling else self.num_samples // 2
        sampled_sizes = self.gen.choice(valid_sizes, num_sizes, p=prob_sizes)
        for idx, s in enumerate(sampled_sizes):
            indices = self.gen.choice(self.n, s, replace=False)
            # weight = Pr(sampling this set) * w(s)
            weight = 1 / (self.sample_weight(s) * s * (self.n - s))
            self.add_one_sample(idx, indices, weight=weight)
    
    def find_constant_for_bernoulli(self, max_C = 1e10):
        # Choose C so that sampling without replacement from min(1, C*prob) gives the same expected number of samples
        C = 1 # Assume at least n - 1 samples
        m = min(self.num_samples, 2**self.n) # Maximum number of samples is 2^n
        def expected_samples(C):
            expected = [min(scipy.special.binom(self.n, s), 2* C * self.sample_weight(s)) for s in range(self.n+1)]
            return np.sum(expected)
        # Efficiently find C with binary search
        L = 1
        # Compute smallest probability
        R = 1/min(self.sample_weight(s) for s in range(self.n)) * 2**self.n
        while round(expected_samples(C)) != m:
            #print(f'Expected samples: {expected_samples(C)}')
            #print(f'Constraint: {m}')
            #print(f'C: {C}')
            if expected_samples(C) < m: L = C
            else: R = C
            C = (L + R) / 2
        self.C = round(C)
    
    def sample_without_replacement(self):
        self.find_constant_for_bernoulli()
        m_s_all = []
        for s in range(self.n):
            # Sample from Binomial distribution with (n choose s) trials and probability min(1, C*sample_weight(s) / (n choose s))
            prob = min(1, 2*self.C * self.sample_weight(s) / scipy.special.binom(self.n, s))
            try:
                m_s = self.gen.binomial(int(scipy.special.binom(self.n, s)), prob)
            except OverflowError: # If the number of samples is too large, assume the number of samples is the expected number
                m_s = int(prob * scipy.special.binom(self.n, s))
            if self.paired_sampling:
                if s == self.n // 2: # Already sampled all larger sets with the complement                    
                    if self.n % 2 == 0: # Special handling for middle set size if n is even
                        m_s_all.append(m_s // 2)
                    else: m_s_all.append(m_s)
                    break
            m_s_all.append(m_s)
        sampled_m = np.sum(m_s_all)
        num_rows = sampled_m if not self.paired_sampling else sampled_m * 2
        self.A_tilde = np.zeros((num_rows, self.n))
        idx = 0
        for s, m_s in enumerate(m_s_all):
            prob = min(1, 2*self.C * self.sample_weight(s) / scipy.special.binom(self.n, s))
            weight = 1 / prob
            if self.paired_sampling and s == self.n // 2 and self.n % 2 == 0:
                # Partition the all middle sets into two
                # based on whether the combination contains n-1
                combo_gen = combination_generator(self.gen, self.n - 1, s-1, m_s)
                for indices in combo_gen:
                    self.add_one_sample(idx, list(indices) + [self.n-1], weight = weight)
                    idx += 1
            else:
                combo_gen = combination_generator(self.gen, self.n, s, m_s)
                for indices in combo_gen:
                    self.add_one_sample(idx, list(indices), weight = weight)
                    idx += 1
    
    def compute(self):
        # Sample
        self.sample()
        # x* = (A^T A)^-1 A^T v
        # xt = (A^T S^T W S A)^-1 A^T S^T W S v
        # phit = (1+bn 1 1^T) xt
        A_tilde = self.A_tilde / self.an

        binary_tilde = (A_tilde > 0).astype(int)
#        print('A_tilde:', A_tilde)
#        print('binary_tilde:', binary_tilde)

        inputs = self.baseline * (1 - binary_tilde) + self.explicand * binary_tilde
        Sv = self.model.predict(inputs) 
#        print('kernel weights:', self.kernel_weights)

        ASSv = A_tilde.T @ np.diag(self.kernel_weights) @ Sv
        ASSA = A_tilde.T @ np.diag(self.kernel_weights) @ A_tilde
#        print('ASSA', ASSA)
#        print('ASSv', ASSv)
        x_star = np.linalg.lstsq(ASSA, ASSv, rcond=None)[0]

        ATA = np.eye(self.n) + self.bn * np.ones((self.n, self.n))
#        print('ATA:', ATA)

        self.phi = ATA @ x_star

        return self.phi

def kernel_prob(baseline, explicand, model, num_samples, p):
    estimator = ProbRegressionEstimator(model, baseline, explicand, num_samples, p=p, paired_sampling=True, leverage_sampling=True, bernoulli_sampling=True)
    return estimator.compute()

import numpy as np

def e(y, n_clases):
    canonical_vector = []
    for i in range(len(y)):
        zeros_before = np.zeros(y[i]-1)
        zeros_after = np.zeros(n_clases-y[i])
        one = np.array([1])
        row = np.concatenate((zeros_before, one, zeros_after))
        canonical_vector.append(row)
    return np.array(canonical_vector)
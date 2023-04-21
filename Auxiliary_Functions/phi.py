import numpy as np

def phi(Mdl, x, y):
    if Mdl.feature_map == 'linear':
        # Linear Kernel
        linear = x
        if Mdl.intercep == True:
            map = np.kron(e(y, Mdl.labels), np.concatenate(([1], linear)))
        if Mdl.intercep == False:
            map = np.kron(e(y, Mdl.labels), linear)

    elif Mdl.feature_map == 'polinom':
        # Polynomial Kernel (degree 2)
        pol = np.concatenate((x, x ** 2), axis=1)
        if Mdl.intercep == True:
            map = np.kron(e(y, Mdl.labels), np.concatenate(([1], pol), axis=1))
        if Mdl.intercep == False:
            map = np.kron(e(y, Mdl.labels), pol)

    elif Mdl.feature_map == 'polinom3':
        # Polynomial Kernel (degree 3)
        pol = np.concatenate((x, x ** 2, x ** 3), axis=1)
        if Mdl.intercep == True:
            map = np.kron(e(y, Mdl.labels), np.concatenate(([1], pol), axis=1))
        if Mdl.intercep == False:
            map = np.kron(e(y, Mdl.labels), pol)

    elif Mdl.feature_map == 'random':
        # Random Feature
        random_cos = []
        random_sin = []
        for i in range(Mdl.u.shape[1]):
            random_cos = np.concatenate((random_cos, np.cos(np.dot(x, Mdl.u[:, i]))), axis=1)
            random_sin = np.concatenate((random_sin, np.sin(np.dot(x, Mdl.u[:, i]))), axis=1)
        random = np.sqrt(1 / Mdl.u.shape[1]) * np.concatenate((random_cos, random_sin), axis=1)
        if Mdl.intercep == True:
            map = np.kron(e(y, Mdl.labels), np.concatenate(([1], random), axis=1))
        if Mdl.intercep == False:
            map = np.kron(e(y, Mdl.labels), random)

    elif Mdl.feature_map == 'indicatriz':
        indicatriz = np.concatenate((x[:, 1] * (x[:, 0] >= 0), x[:, 1] * (x[:, 0] < 0)), axis=1)
        if Mdl.intercep == True:
            map = np.kron(e(y, Mdl.labels), np.concatenate(([1], indicatriz), axis=1))
        if Mdl.intercep == False:
            map = np.kron(e(y, Mdl.labels), indicatriz)

    return map

def e(y, n_clases):
    canonical_vector = []
    for i in range(len(y)):
        zeros_before = np.zeros(y[i]-1)
        zeros_after = np.zeros(n_clases-y[i])
        one = np.array([1])
        row = np.concatenate((zeros_before, one, zeros_after))
        canonical_vector.append(row)
    return np.array(canonical_vector)
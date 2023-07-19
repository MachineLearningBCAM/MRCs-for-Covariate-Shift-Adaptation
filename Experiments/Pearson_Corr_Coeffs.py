import numpy as np

def PCC(Xtr, Xte, Ytr, Yte, D):
    X = np.vstack((Xtr, Xte))
    Y = np.vstack((Ytr, Yte))

    m = X.shape[1]

    pearson_coeffs = np.zeros(m)
    for i in range(m):
        pearson_coeffs[i] = np.corrcoef(X[:, i], Y, rowvar=False)[0, 1]

    pearson_coeffs = pearson_coeffs[~np.isnan(pearson_coeffs)]
    sorted_indices = np.argsort(np.abs(pearson_coeffs))[::-1]

    top_features = sorted_indices[:D]

    Xtr = Xtr[:, top_features]
    Xte = Xte[:, top_features]

    return Xtr, Xte
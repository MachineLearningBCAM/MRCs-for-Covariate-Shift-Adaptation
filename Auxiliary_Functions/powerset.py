import numpy as np
from itertools import combinations

def powerset(n):
    set = []
    for r in range(1, n+1):
        for combo in combinations(range(1, n+1), r):
            set.append(np.array(combo))
    return set

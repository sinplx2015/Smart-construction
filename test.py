from copy import deepcopy
import numpy as np
from math import sqrt

def svd(A, eps=1e-10):
    mat = deepcopy(A)
    sigma = np.zeros_like(A).astype(float)
    values, vectors = np.linalg.eig(mat.T @ mat)
    U = np.linalg.eig(mat @ mat.T)[1]   
    k=0
    for value in values:        
        if value > eps:
            sigma[k,k] = sqrt(value)
            k += 1
    return  U, sigma, vectors.T


E = np.array([[1, 1], [2, 2], [0, 0]]) 
print(svd(E))
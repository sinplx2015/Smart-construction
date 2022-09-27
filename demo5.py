from copy import deepcopy
import numpy as np
from math import sqrt

def FRdecomposition(matrix):
    """
    Full Rank Decomposition
    """
    M, N = np.shape(matrix)
    raw_A = deepcopy(matrix)
    A = deepcopy(matrix)
    for i in range(M):
        for j in range(i, N):
            
            #resort
            if sum(abs(A[i:, j])) != 0:
                # print(sum(abs(A[i:, j])))
                zero_index = []
                non_zero_index = []
                for k in range(i, M):
                    if A[k, j] != 0:
                        non_zero_index.append(k)
                    else:
                        zero_index.append(k)
                non_zero_index = sorted(non_zero_index, key=lambda x : abs(A[x, j]))
                sort_cols = non_zero_index+zero_index
                A[i:, :] = A[sort_cols, :]

                # eliminate elements belows
                prefix = -A[i+1:, j]/A[i, j]
                temp = (np.array(prefix).reshape(-1, 1))@A[i, :].reshape((1, -1))
                A[i+1:, :] += temp         
                # eliminate elements above
                for k in range(i):
                    if A[k, j] != 0:
                        A[k, :] += -(A[k, j]/A[i, j])*A[i, :]
                break

            else:
                continue
    principal_indexs = [[], []]
    for m in range(M):
        for n in range(m, N):
            if A[m, n] != 0:
                principal_indexs[0].append(n)
                principal_indexs[1].append(m)
                #normalize
                if A[m, n] != 1:
                    A[m, :] *= 1./A[m, n]
                break
    
    if (len(principal_indexs[0])):
        F = raw_A[:, principal_indexs[0]]
        G = A[principal_indexs[1], :]
        return F, G
    else:
        Exception("-1")

def qr(A):
    """
    QR alg
    """
    n = A.shape[1]
    Q = np.zeros_like(A)
    R = np.zeros((n,n))
    for j in range(n):
        v = A[:, j]
        for i in range(j ):
            q = Q[:, i]
            R[i, j] = q @ v
            # print(R[i, j])
            v = v - R[i, j] * q            
        norm = np.linalg.norm(v)
        Q[:, j] = v / norm
        R[j, j] = norm
    return Q, R

def svd(A, eps=1e-10):
    """
    Singular Value Decomposition
    Give an A, return U, Sigma and V^H
    """
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

def main():
    A = np.array([[2,0,1,4], [0,1,0,2], [2,-1,1,2]]).reshape((3,4)).astype(float)
    F,G=FRdecomposition(A)
    print('1.\n',F,'\n',G)

    B = np.array([[1,1,2], [1,2,1], [1,1,3], [2,3,3]]).astype(float)
    Q,R=qr(B)
    print('2.\n',Q,'\n',R)
    C = np.array([[1, 1], [2, 2], [0, 0]]) 
    print('3.\n', svd(C)[0], '\n', svd(C)[1], '\n', svd(C)[2])
if __name__ == '__main__':
    main()

import numpy as np
import math
import sympy
from demo2 import uppertriangular, matmul, get_col, get_row, uppertriangular

def norm(a):
    return math.sqrt(sum([x_i**2 for x_i in a]))

def qr(A):
    """
    QR Decomposition
    """
    A = np.array(A,dtype='float')
    m, n = A.shape
    Q = np.eye(m)
    for i in range(n - (m == n)):
        H = np.eye(m)
        H[i:, i:] = make_householder(A[i:, i])
        Q = np.array(matmul(Q ,H))
        A = np.array(matmul(H ,A))
    return Q, A

def make_householder(a):
    """
    Make Householder Transformer
    """
    u = a / (a[0] + np.copysign(norm(a), a[0]))
    u[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(u, u)) * np.array(matmul(u[:, None], u[None, :]))
    return H

def QR_eigvalues(A, tol=1e-15, maxiter=1000):
    A_new = A
    i=0
    diff = np.inf
    while (diff > tol) and (i < maxiter):
    # while i < maxiter:
        A_old= A_new
        Q, R = qr(A_old)
        A_new= np.array(matmul(R, Q))
        diff = np.abs(A_new - A_old).max()
        i += 1
    eigvalues = np.diag(A_new)
    return eigvalues

# def dot(L , U,  m, n):
#     return (sum(L[m][i]*U[i][n] for i in range(m)))

# def doolittle(A): 
#     A =  A = np.array(A,dtype='float')  
#     n = get_row(A)    
#     U = np.zeros((n, n))
#     L = np.eye(n)
    
#     for k in range(n):
#         L[k, k] = 1
#         U[k, k] = A[k, k] - (sum(L[k][x]*U[x][k] for x in range(k)))
#         for j in range(k+1, n):
#             U[k, j] = A[k, j] -(sum(L[k][x]*U[x][j] for x in range(k)))
#         for i in range(k+1, n):
#             L[i, k] = (A[i, k] - (sum(L[i][x]*U[x][k] for x in range(k)))) / U[k, k]
    
#     return L, U

# def solve(A):
#     A = np.array(A)
#     n = get_row(A)
#     b = ([])
#     L, U =doolittle(A)
#     for i in range (n):
#         b.extend([[0]])  

#     y = np.zeros_like(b, dtype=np.double)
#     y[0] = b[0] / A[0, 0]
    
#     for i in range(1, n):
#         y[i] = (b[i] - np.dot(A[i,:i], y[:i])) / A[i,i]

#     x = np.zeros_like(y, dtype=np.double);
#     x[-1] = y[-1] / U[-1, -1]
#     for i in range(n-2, -1, -1):
#         x[i] = (y[i] - np.dot(U[i,i:], x[i:])) / U[i,i]
        
#     return x

# def LUsolver(A):
#     m = get_row(A)
#     b = ([])
#     for i in range (m):
#         b.extend([[0]])
#     x = np.zeros(m) 
#     y = np.zeros(m) 
#     L, U = doolittle(A)

#     for i in range(0, m, 1):
#         y[i] = b[i] / L[i][i]
#         for j in range(0, i, 1):
#             y[i] -= y[i] * L[i][j]

#     for i in range(m-1, -1, -1):
#         x[i] = y[i] / U[i][i]
#         for j in range(i-1, -1, -1):
#             U[i] -= x[i] * U[i][j]
#     return x
from sympy import *
from sympy.abc import x,y,z

def eigenvector(A):
    #Only for 3x3
    values = QR_eigvalues(A)
    res = []
    for value in values:
        print(value)
        A_0 = np.copy(A)
        for j in range(get_row(A)):
            
            A_0[j][j] -= round(value,10)
        print(A_0)
        for i in range(get_row(A)):
            for j in range (get_col(A)): 
                A_0[i][j]=round(A_0[i][j])
        eq1 = A_0[0][0] * x + A_0[0][1] * y + A_0[0][2] *z
        eq2 = A_0[1][0] * x + A_0[1][1] * y + A_0[1][2] *z
        eq3 = A_0[2][0] * x + A_0[2][1] * y + A_0[2][2] *z

        ans = sympy.solve([eq1, eq2, eq3], [x, y, z]) 
        res.append(ans)
        
    return  res

A = ([[3, 5, 5], [5, 3, 5], [-5, -5, -7]]) 
print(QR_eigvalues(A))
print(eigenvector(A))

from copy import deepcopy
from webbrowser import get
from demo import get_col, get_row, matmul
import numpy as np
import math
import sympy
from sympy.abc import x,y,z

#utils
def swap_row(A,i,j):
    """
    Swap 2 rows
    """
    A[i], A[j] = A[j], A[i]
    return A

def trans(A):
    """
    Transpose a Matrix
    """
    res = []
    A_copy = deepcopy(A)
    for i in range (get_col(A)):
        B = [0]*get_row(A)
        for j in range(get_row(A)):
            A_copy = deepcopy(A)
            B[j] = A_copy[j][i]
        res.append(B)
    
    return res
def eliminate(list1, list2, num, target = 0):
    """
    Eliminate LowerTriangular Elements in the Same Col 
    """
    alpha = (list2[num]-target) / list1[num]
    for i in range(len(list2)):
        list2[i] -= alpha * list1[i]

def gauss(A):
    """
    Gauss Elininate Method
    """
    for i in range(get_row(A)):
        
        if A[i][i] == 0:
            for j in range(i+1, get_row(A)):
                if A[i][j] != 0:
                    swap_row(A, i ,j)
                    break
            else:
                raise Exception("-1")

        for j in range(i+1, get_row(A)):
            eliminate(A[i], A[j], i)

    for i in range(len(A)-1, -1, -1):
        for j in range(i-1, -1, -1):
            eliminate(A[i], A[j], i)

    for i in range(get_row(A)):
        eliminate(A[i], A[i], i, target=1)        
    return A



def uppertriangular(A):  
    """
    Convert to UpperTriangular Matrix
    """
    m = get_row(A)
    n = get_col(A)
    A_new = A 
    for num in range(n): 
        for i in range(num+1,m): 

            if A_new[num][num] == 0 :
                for k in range (i,m):
                    if A_new[k][num] != 0: 
                        swap_row(A_new,num,k)
                        break
                    elif A_new[k] != [0]*n :
                        swap_row(A_new,num,k)
                        return A_new
                    return A_new 

            eliminate(A_new[num],A_new[i],num)
            
    return A_new

def inverse(A):
    """
    Get Inverse Matrix
    """
    tmp = [[] for _ in A]
    for i,row in enumerate(A):
        assert len(row) == get_row(A)
        tmp[i].extend(row + [0]*i + [1] + [0]*(get_row(A)-i-1))
    gauss(tmp)
    res = []
    for i in range(get_row(tmp)):
        res.append(tmp[i][len(tmp[i])//2:])
    return res


def rank(A):
    """
    Get Rank of A Matrix
    """
    m = get_row(A)
    n = get_col(A)
    M = uppertriangular (A)
    M = trans(M)
    M = uppertriangular(M)
    
    for i in range(n):
        for j in range (m): 
            M[i][j]=round(M[i][j])    

    for k in range(n):
        if M[k] == [0]*m:
            return k
    return m



"""
Eigenvalues and Eigenvectors
"""
# def norm(a):
    
#     return math.sqrt(sum([x_i**2 for x_i in a]))

def qr(A):
    """
    QR alg
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
    Make householder
    """
    a = np.array(a,dtype='float')
    u = a / (a[0] + np.copysign(np.linalg.norm(a), a[0]))
    u[0] = 1
    H = np.eye(a.shape[0])
    H -= (2 / np.dot(u, u)) * np.array(matmul(u[:, None], u[None, :]))
    return H

def QR_eigvalues(A, tol=1e-15, maxiter=1000):
    """
    Compute Eigenvalues from QR iter
    """
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


def eigenvector(A):
    """
    Just for 3*3
    """
    values = QR_eigvalues(A)
    vectors = []
    for value in values:

        A_0 = np.copy(A)
        for j in range(get_row(A)):
            A_0[j][j] -= round(value,10)

        for i in range(get_row(A)):
            for j in range (get_col(A)): 
                A_0[i][j]=round(A_0[i][j])

        eq1 = A_0[0][0] * x + A_0[0][1] * y + A_0[0][2] *z
        eq2 = A_0[1][0] * x + A_0[1][1] * y + A_0[1][2] *z
        eq3 = A_0[2][0] * x + A_0[2][1] * y + A_0[2][2] *z

        x_expr, y_expr, z_expr = sympy.solve([eq1, eq2, eq3], [x, y, z],force=True, manual=True)[0]

        count = 0
        for i in range(get_row(A)):
            if round(values[i]) == round(value):
                count += 1
                
        if count == 1  :
            value_x = x_expr.subs([(x, 1), (y, 1), (z, 1)])
            value_y = y_expr.subs([(x, 1), (y, 1), (z, 1)])
            value_z = z_expr.subs([(x, 1), (y, 1), (z, 1)])
            vectors.append([value_x, value_y, value_z])
        else:
            if x_expr == 0: subs = [[(y, 1), (z, 1)], [(y, -1), (z, -1)]]
            elif y_expr == 0: subs = [[(z, 1)], [(z, -1)]]
            elif z_expr == 0: subs = [[(y, 1)], [(y, -1)]]
            else: subs = [[(y, 0), (z, 1)], [(y, 1), (z, 0)]]
            n_1 =[x_expr.subs(subs[0]),y_expr.subs(subs[0]),z_expr.subs(subs[0])]
            n_2 =[x_expr.subs(subs[1]),y_expr.subs(subs[1]),z_expr.subs(subs[1])]
            if n_1 not in vectors and n_2 not in vectors:
                vectors.append(n_1)
                vectors.append(n_2)
        
        # res.append(ans)        
    return  vectors




def main():
    A = ([[0, 4, 10, 1], [4, 8, 18, 7], [10, 18, 40, 17], [1, 7, 17, 3]])
    B = ([[1, 2, 3, 4], [2, 3, 1, 2], [1, 1, 1, -1], [2, 1, -1, -7]])
    C = ([[3, -10, 3, -2, -4], [-1, 4, -2, 0, 3], [-4, 22, -14, -1, 23], [2, -10, 1, -8, -3]])
    D = ([[3, 5, 5], [5, 3, 5], [-5, -5, -7]]) 
    E = ([[1, 2, 4], [2, -2, 2], [4, 2, 1]]) 
    print('1.:\n', uppertriangular(A))
    print('2.:\n', inverse(B))
    print('3.:\n', rank(C))
    print('4.:\n',QR_eigvalues(D),eigenvector(D))
    print('5.:\n',qr(eigenvector(E))[0])

if __name__ == "__main__":
    main()
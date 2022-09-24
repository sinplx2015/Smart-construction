from copy import deepcopy
from demo import get_col, get_row, matmul


def swap_row(A,i,j):
    """
    Swap 2 rows
    """
    A[i], A[j] = A[j], A[i]
    return A
    
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


def main():
    A = ([[1,1,1,1],[1,1,-1,-1],[1,-1,1,-1],[1,-1,-1,1]])
    B = ([[1,2,1,0],[1,1,1,1],[0,2,1,-1],[1,1,0,-1]])
    T = matmul(inverse(A),B)
    print('1.:\n' ,T)
    x = ([[1],[0],[0],[-1]])
    print('2.:\n',matmul(inverse(T),x))
    
if __name__ == "__main__":
    
    main()

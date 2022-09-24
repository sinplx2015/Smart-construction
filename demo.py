def get_row(A):
    """
    得到矩阵行数
    row = 1 if isinstance(A[0],(int, float)) else len(A)
    """
    return row

def get_col(A):
    """
    得到矩阵列数
    """
    col = len(A) if isinstance(A[0], (int, float)) else len(A[0])
    
    
    return col 

def matmul(A,B):
    """
    计算矩阵乘积
    """
    if get_row(B) != get_col(A):
        raise Exception("-1")
    if get_row(A) == 1 :
        res = [sum(a * b for a, b in zip(A, B_col))
        for B_col in zip(*B)]
    else:
        res = [[sum(a * b for a, b in zip(A_row, B_col))
        for B_col in zip(*B)] for A_row in A]
    return res

def det(A):
    """
    先将矩阵化为上三角，再计算行列式
    """
    if get_row(A) != get_col(A):
        raise Exception("-1")
    res = 1.0
    n = len(A)
    A_new = A    
    for num in range(n): 
        for i in range(num+1,n): 
            if A_new[num][num] != 0:
                alpha = A_new[i][num] / A_new[num][num]    
                for j in range(n): 
                    A_new[i][j] = A_new[i][j] - alpha * A_new[num][j] 
            else:
                A_new[num][num] = 1.0e-10
            
    for i in range(n): 
        res *= A_new[i][i] 
    return res


def main():
    #EX 1.(1)
    A1 = ([[2, 5, 7], [5, -2, -3], [6, 3, 4]])
    B1 = ([[5],[1],[0.5]])
    print('1.(1):\n', matmul(A1,B1))

    #EX 1.(2)
    A2 = ([5, 1, 0.5])
    B2 = ([[2, 5, 7], [5, -2, -3], [6, 3, 4]])
    print('1.(2):', matmul(A2,B2))

    #EX 2.
    A3 = ([[1, -1, 3], [-2, 3, -11], [4, -5, 17]])
    B3 = ([[4, 7, -3], [-2, -4, 2], [4, -10, 4]])
    print('2:\n', matmul(A3,B3))

    #EX 3.
    A4 = ([[1, 2, 3], [4, 5, -4], [-3, -2, -1]])
    print('3:\n',"%.1f" %det(A4))

if __name__ == "__main__":
    main()
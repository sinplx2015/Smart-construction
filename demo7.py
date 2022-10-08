import sympy
from scipy.optimize import minimize

x, y, lmd = sympy.symbols('x, y, lmd')
L = x**2 + y**2 - lmd * (x * y - 3)

dif1 = sympy.diff(L, x)
dif2 = sympy.diff(L, y)
dif3 = sympy.diff(L, lmd)

res = sympy.solve([dif1, dif2, dif3], [x, y, lmd])
print(res)



# Objective function
fun = lambda x: (x[0]-1)**2 + (x[1]-2)**2

# constraints
cons = ({'type': 'eq', 'fun': lambda x: x[0]-x[1] + 1},
        {'type': 'ineq', 'fun': lambda x: x[0]+x[1] - 2},
        {'type': 'ineq', 'fun': lambda x: -x[0]},
        {'type': 'ineq', 'fun': lambda x: -x[1]}
        )

bnds = ((None, None), )*2

# initial guesses
x0s = [[1,2],
    #    [2,3],
    #    [3,4],
    #    [4,5],
    ]

for x0 in x0s:
    print(x0)
    res1 = minimize(fun, x0, method='SLSQP', bounds=bnds, constraints=cons)
    print('\n',res1)
    print("optimal value p*", res1.fun)
    print("optimal var: x1 = ", res1.x[0], " x2 = ", res1.x[1])
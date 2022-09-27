import sympy 

#2
r, theta, phi  = sympy.symbols('r, theta, phi')
F = sympy.Matrix([
    r * sympy.sin(theta) * sympy.cos(phi), 
    r * sympy.sin(theta) * sympy.sin(phi),
    r * sympy.cos(theta)
])
args =([r, theta, phi])
print('2:\n',F.jacobian(args))

#3
a, b, c, y, x = sympy.symbols('a, b, c, y, x')
y = sympy.Matrix([3 * a - 2 * b + 4 * c])
args =([a], [b], [c])
print('3:\n',y.jacobian(args).T)

#4
x_1, x_2, x_3, f = sympy.symbols('x_1, x_2, x_3, f')
f = sympy.Matrix([x_1**2+2*(x_2**2)-x_3**2+4*x_1*x_2-4*x_1*x_3-4*x_2*x_3 ])
args = ([x_1, x_2, x_3])
print('4:\n',f.jacobian(args).T)
# LINEAR ALGEBRA

# Dot Product

def dot_product(x, y):
    return sum(i*j for i, j in zip(x, y))

dot_product([3, 2, 6], [1, 7, -2])


# --------------------------------------------------------------------

#  Norm of a Vector

import math

def norm_verctor(v):
    dot_product = sum(i * i for i in v)
    return math.sqrt(dot_product)
norm_verctor([-1, -2, 3, 4, 5])

# --------------------------------------------------------------------

# Matrix Addition

def matrix_add(x,y):
    xrows = len(x)
    xcols = len(x[0])
    yrows = len(y)
    ycols = len(y[0])
    if xrows != yrows or xcols != ycols:
        print('Can not sum matrixes')
    else:
        z = x
        for i in range(xrows):
            for j in range(xcols):
                z[i][j] = z[i][j] + y[i][j]
    return z

matrix_add([[1,2,5], [3,4,1]], [[5,1,2], [9,3,4]])


# --------------------------------------------------------------------

# Scalar Multiplication

def scalar_mult(c, X):
    cX = X
    for i in range(len(X)):
        for j in range(len(X[0])):
            cX[i][j] = c*cX[i][j]
    return cX

scalar_mult(-3, [[2,6,-1], [2,8,0], [9,8,7]])

# --------------------------------------------------------------------

# Matrix Subtraction

def matrix_sub(x,y):
    xrows = len(x)
    xcols = len(x[0])
    yrows = len(y)
    ycols = len(y[0])
    if xrows != yrows or xcols != ycols:
        print('Can not subtract matrixes')
    else:
        z = x
        for i in range(xrows):
            for j in range(xcols):
                z[i][j] = z[i][j] - y[i][j]
    return z
matrix_sub([[5,2,3], [3,4,-9]], [[5,3,2], [8,2,4]])

# --------------------------------------------------------------------

# Matrix Miltiplication

def matrix_mult(x,y):
    xrows = len(x)
    xcols = len(x[0])
    yrows = len(y)
    ycols = len(y[0])
    if xcols != yrows:
        print('Cannot multiply matrixes')
    else:
        z = [[0 for i in range(ycols)] for j in range(xrows)]
        for i in range(xrows):
            for j in range(ycols):
                total = 0
                for ii in range(ycols):
                    total += x[i][ii] * y[ii][j]
                z[i][j] = total
        return z

matrix_mult([[1,2,5], [3,4,1]], [[5,1,2], [9,3,4], [1,5,3]])


# --------------------------------------------------------------------

# Matrix Transposed

def matrix_transp(x):
    xrows = len(x)
    xcols = len(x[0])
    z = [[0 for i in range(xrows)] for j in range(xcols)]
    for i in range(xcols):
        for j in range(xrows):
            z[i][j] = x[j][i]
    return z


matrix_transp([[1, 2, 5], [3, 5, 4]])

import numpy as np
# !pip install scikit-learnt
# !pip install matplotlib
# !pip install seaborn
# !pip install scipy
# !pip install statsmodels
# !pip install yfinance

new_array = np.array([1, 2, 3, 4])

# ARRAYS

print(new_array)
print('\n')

print(f'{np.zeros([4, 5])} \n')

print(f'{np.ones([4, 5])} \n')

print(np.empty([2,3]))
print('\n')

print(np.arange(15))
a = np.arange(15)
print('\n')

print(a.reshape(3,5))
print('\n')

print(np.linspace(0,30,6))
print('\n')

print(np.arange(27).reshape(3,3,3))
print('\n')


# ARITHMETIC OPPERATIONS WITH ARRAYS

a = np.array([[1,2,1],[2,2,3]])
b = np.array([3,4,5])

print(np.add(a,b))

print(np.subtrack(b,a))

print(np.divide(a,b))

print(np.power(a,2)) # or np.power(a,b)

# CONDITIONAL STATEMENTS

a = np.array([i for i in range(10)])

print(np.where(a%2 == 0, 'Even', 'Odd'))

condition = [a<5, a>5]
do_list = [a**2, a**3]
defualt = a
print(np.select(condition, do_list, defualt))

# MATH AND STATS

a = np.array([[4,3,2], [10,1,0], [5,8,24]])

print(np.amin(a))
print(np.amin(a, axis = 0))
print(np.amin(a, axis = 1))
print('\n')
print(np.amax(a))
print(np.amax(a, axis = 0))
print(np.amax(a, axis = 1))
print('\n')
print(np.median(a))
print(np.mean(a))
print(np.var(a))
print(np.std(a))
print(np.percentile(a,50))
print('\n')
print('\n')


deg = np.array([0,30,45,60,90])

print(np.sin(deg*np.pi/100)) # arcsin
print(np.cos(deg*np.pi/100)) # arccos
print(np.tan(deg*np.pi/100)) # arctan
print('\n')
print('\n')


e = np.array([0.1, 0.8, -2.2, -9.87])

print(np.floor(e))
print(np.ceil(e))

# Multiple demintional arrays

a_1D = np.array([1, 2, 3, 4, 5, 6])
a_2D = np.array([[1, 2, 3], [4, 5, 6]])
a_3D = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])

a_1D[2:-5]

a_2D[0,2]
a_2D[0:2:-1]

a_3D[0,1,2]
a_3D[0,1:,2:]

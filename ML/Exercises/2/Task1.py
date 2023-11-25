import numpy as np

x = np.array([[1.5,2] , [3,2.5], [4.5,3]])
y = np.array([10,15.5,21])
theta = np.zeros(2)

x_t = np.transpose(x)
a = x_t@x
b = x_t@y
print(b)
solution = np.linalg.solve(a, b)
print(solution)


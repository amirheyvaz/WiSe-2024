import numpy as np

x = np.array([[1,1] , [2,1], [3,1]])
y = np.array([3,4,6])
theta = np.zeros(2)

x_t = np.transpose(x)
a = x_t@x
b = x_t@y

solution = np.linalg.solve(a, b)
print(solution)


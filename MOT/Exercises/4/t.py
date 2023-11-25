# the code used for stochastic gredient descent with adagrad
import numpy as np
b = np.array([[1], [1]], dtype='float')
x = np.array([[1.5,2], [3,2.5], [4.5,3]])
y = np.array([[10],[15.5],[21]])
moo = 0.1
epochs = 2
epsilon = 1e-8
def grad(x, y,b):
    return 2*(x.T@x@b-x.T@y)

def mse(x,y,b):
    return (x@b - y)**2
gradian_history = np.array([[0],[0]], dtype='float')
for epoch in range(epochs):
    for i in range(len(x)):
        x_i = x[i, :].reshape((1,2))
        y_i = y[i, :].reshape((1,1))
        gradient = -grad(x_i, y_i,b)
        err = mse(x_i, y_i,b)
        print(f'epoch {epoch} - step {i}\n===============================')
        print(f'b: {b}')
        print(f'mse: {err}')
        gradian_history += gradient**2
        step_size = moo / (np.sqrt(gradian_history) + epsilon)
        b = b - step_size * gradient
print(f'final b is : {b}')
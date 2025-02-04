import numpy as np

x = np.array([1, 3, 5, 20, 14, 6, 7])
y = np.array([2000.0, 5600.0, 1400.0, 2500.0, 1245.0, 987.0, 1234.0])

m = x.shape[0]

def cost_func(w_in, b_in):
    total_cost = 0

    for i in range(m):
        total_cost += ((x[i] * w_in + b_in) - y[i]) ** 2

    return total_cost / (2 * m)

def gradient(w_in, b_in):
    dw = 0
    db = 0
    for i in range(m):
        j = (x[i] * w_in + b_in) - y[i]
        dw += j * x[i]
        db += j

    return dw/m, db/m

def gradient_descent(num_iter):
    alpha = 0.003
    w = 100
    b = 10

    for i in range(num_iter):
        dw, db = gradient(w,b) 

        w = w - alpha * dw
        b = b - alpha * db

    return w,b


w,b = gradient_descent(10000)
print(f"Total cost for w = {w} and b = {b} is {cost_func(w,b)}")

import numpy as np

x = np.arange(500).reshape(-1, 10)
y = np.random.rand(x.shape[0])
m = x.shape[0]

def cost(w_in, b_in):
    total_cost = 0.0
    for i in range(m):
        total_cost += ((np.dot(x[i], w_in) + b_in) - y[i]) ** 2

    return total_cost / (2 * m)

def gradient(w_in, b_in):
    dw = np.zeros(w_in.shape)
    db = 0
    for i in range(x.shape[0]):
        err = (np.dot(w_in, x[i]) + b_in) - y[i]
        dw += err * x[i]
        db += err

    return dw/m, db/m

def gradient_descent(num_iter):
    w = np.zeros(x.shape[1])
    b = 0.
    alpha = 0.0000003
    
    for i in range(num_iter):
        dw, db = gradient(w, b)

        w = w - alpha * dw
        b = b - alpha * db

        if i % 10 == 0:
            print(f"[{i}]: cost={cost(w,b)}")

    return w,b


w,b = gradient_descent(1000)
import numpy as np

x = np.arange(500).reshape(-1, 10)
y = np.random.rand(x.shape[0]).reshape(-1, 1)
m = x.shape[0]

def cost(w_in, b_in):
    total_cost = ((x @ w_in + b_in) - y) ** 2
    total_cost = np.sum(total_cost)

    return total_cost / (2 * m)

def gradient(w_in, b_in):
    err = (x @ w_in + b_in) - y
    
    dw = x.transpose() @ err
    db = np.sum(err)

    return dw/m, db/m

def gradient_descent(num_iter):
    w = np.zeros(x.shape[1]).reshape(-1,1)
    b = 0.
    alpha = 0.000001
    
    for i in range(num_iter):
        dw, db = gradient(w, b)

        w = w - alpha * dw
        b = b - alpha * db

        if i % 10 == 0:
            print(f"[{i}]: cost={cost(w,b)}")

    return w,b


w,b = gradient_descent(1000)
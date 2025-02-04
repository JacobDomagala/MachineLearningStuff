import numpy as np
import matplotlib.pyplot as plt

x = np.arange(500, dtype=np.float64).reshape(-1, 10)

# Scale x
avg = np.transpose(np.average(x, axis=0))

max = np.max(x, axis=0)
min = np.min(x, axis=0)
res = np.transpose(max - min)
x = (x - avg) / res

# scaled_x = (x @ (avg*-1)) @ (max - min) * 
print(f"{x}")

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

def gradient_descent(num_iter, alpha):
    J = []
    w = np.zeros(x.shape[1]).reshape(-1,1)
    b = 0.
    
    for i in range(num_iter):
        dw, db = gradient(w, b)

        w = w - alpha * dw
        b = b - alpha * db

        cost_i = cost(w,b)
        if i % 10 == 0:
            print(f"[{i}]: cost={cost_i}")
        
        J.append(cost_i)

    return w,b,J


w,b,J = gradient_descent(1000, 0.01)

fig, ax = plt.subplots()

ax.plot(range(len(J)), J, linewidth=2.0)

plt.show()

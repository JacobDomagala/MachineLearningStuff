import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  
y = np.array([0, 0, 0, 1, 1, 1])                                           
m = y.shape[0]

def cost(w_in, b_in):
    # make sure w_in is a column vector with proper size
    assert(w_in.shape == (X.shape[1],1))

    z = X @ w_in + b_in

    e = 1 / (1 + np.exp(-z))
    e = e.flatten()

    total_cost = -np.sum(y * np.log(e) + (1 - y) * np.log(1 - e))/m    
        
    return total_cost


print(f"Cost={cost(np.array([0,1]).reshape(-1, 1), 1)}")
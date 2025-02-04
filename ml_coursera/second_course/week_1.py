import numpy as np
import pandas as pd

X = pd.read_csv("employee_attrition.csv").to_numpy()[:, :-1]

W1 = np.random.random((3, 7))
b1 = np.random.random(3)

W2 = np.random.random((6, 3))
b2 = np.random.random(6)

W3 = np.random.random((3, 6))
b3 = np.random.random(3)

W4 = np.random.random((1, 3))
b4 = np.random.random(1)

print(f"shape={X.shape}")

# In our case it's sigmoid for logistic regression
def activation_func(z):
    return 1 / (1 + np.exp(-z))

def compute_layer(a, W, b):
    aT = a.reshape(-1,1)
    bT = b.reshape(-1,1)

    z = W @ aT + bT
    a_new = activation_func(z)
    print(f"Compute layer a={a.shape} aT={aT.shape} w={W.shape} b={b.shape} out={a_new.shape}")
    
    return a_new



a1 = compute_layer(X[:1], W1, b1)
a2 = compute_layer(a1, W2, b2)
a3 = compute_layer(a2, W3, b3)
a4 = compute_layer(a3, W4, b4)

print(a4)
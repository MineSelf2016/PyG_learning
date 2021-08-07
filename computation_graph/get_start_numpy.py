#%%
import torch 
import numpy as np 

# %%
n_samples = 100
n_features = 4
X = np.random.rand(n_samples, n_features)
y = np.random.rand(n_samples)

#%%
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# %%
# n_hidden = 10
learning_rate = 0.1

w = np.random.rand(n_features, 1)

def forward(w, X):
    # Z = np.dot(w.T, X.T)
    A = np.dot(w.T, X.T)
    # A = sigmoid(Z)
    return A

def get_loss(A, y):
    loss = np.sum((A - y) ** 2 / (2 *len(y)))
    return loss

def get_grad(A, y):
    err = A - y
    grad_w = np.dot(err, X) / len(y)
    return grad_w

def backward(w, grad_w, learning_rate):
    w = w - learning_rate * grad_w
    return w 

#%%
w = np.random.rand(n_features, 1)
for i in range(100):
    A = forward(w, X)
    loss = get_loss(A, y)
    print(loss)
    grad_w = get_grad(A, y)
    w = backward(w, grad_w, learning_rate)
    

# %%

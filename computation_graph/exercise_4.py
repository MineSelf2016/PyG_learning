#%%
import torch
import random
import numpy as np 
import os

seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

# %%
x = torch.randn(100, 4)
target = torch.randn(100, 1)
target = target.reshape(1, 100)

# %%
target

# %%
w = torch.randn(4, 1, requires_grad = True)
w

# %%
for epoch in range(20):
    y_pred = torch.mm(w.T, x.T)
    loss = ((y_pred - target)**2).sum() / 100
    print(loss)
    loss.backward()
    w.data = w.data - 0.005 * w.grad.data
    w.grad.data.zero_

# %%
import torch

# %%
x = torch.randn(4, 4)
y = torch.randn(4, 4)

x, y

# %%
x.mm(y)

# %%
x @ y

# %%
x * y

# %%
x.mul(y)

# %%
t = torch.randn(4)
x = t.numpy()

# %%
x

# %%
type(x)

# %%
import numpy as np 
x = np.ones([4, 4])
x

# %%
t = torch.from_numpy(x)
t

# %%

#%%
import torch
from torch import nn 
import numpy as np 

# %%
num_samples = 100
num_features = 4
num_categories = 10

x = torch.randn(num_samples, num_features)
y = np.random.randint(0, num_categories, num_samples)
y = torch.from_numpy(y)
y = torch.zeros(num_samples, num_categories).scatter_(dim = 1, index = y.unsqueeze(1), value = 1)

x.shape, y.shape

# %%
output = net(x)
output.shape

#%%
pred = net.predict(output)
pred

# %%

#%%
import torch
from torch import nn 
import numpy as np 

# %%
x = torch.tensor([[1, 2, 3, 4], [2, 3, 1, 1]], dtype = torch.double)
x


# %%
m = nn.Softmax(dim = 1)
pred_probability = m(x)
pred_probability

# %%
pred_probability.argmax(dim = 1)

#%%

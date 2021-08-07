#%%
import torch 
from torch_scatter import scatter, segment_coo

#%%
x = torch.randn(6, 8)
index = torch.tensor([0, 0, 1, 0, 2, 2])

output = scatter(x, index, dim = 0)
output

# %%

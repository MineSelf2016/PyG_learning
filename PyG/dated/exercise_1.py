#%%
import torch
from torch_geometric.data import Data

# %%
x = torch.tensor([[0, 1], [-1, 0.5], [2, -3.1]], dtype = torch.float)
y = [[0], [0], [1]]

edge_index = [
    [0, 1],
    [1, 0],
    [2, 1],
    [1, 2]
    ]
edge_index = torch.tensor(edge_index).T

graph = Data(x = x, edge_index = edge_index, y = y, color = "red")

# %%
graph

# %%
graph.y

# %%

#%%
import torch 
from torch_geometric.data import Data 
from torch_geometric.data import DataLoader
from torch_scatter import scatter

# %%
edge_index = torch.tensor([
    [0, 1],
    [1, 0],
    [1, 2],
    [2, 1]
])

x = torch.tensor([
    [0, -1],
    [0.5, 2],
    [3, 1]
])

y = [
    [0],
    [1],
    [0]
]

graph = Data(x = x, edge_index = edge_index.T, y = y)

graph

# %%
graph.edge_index

# %%
dataset = [graph, graph, graph]

loader = DataLoader(dataset, batch_size = 2)

# %%
batch = next(iter(loader))

# %%
batch.batch

# %%
batch.edge_index

#%%
batch.x.shape

# %%
output = scatter(batch.x, batch.batch, dim = 0, reduce = "max")
output.shape

# %%
from torch_geometric.datasets import KarateClub
graph = KarateClub()[0]

graph

# %%

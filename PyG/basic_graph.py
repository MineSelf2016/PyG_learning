#%%
import torch
from torch_geometric.data import Data

# %%
edge_index = torch.tensor([
    [0, 1],
    [1, 0],
    [1, 2],
    [2, 1],
    [0, 0]
], dtype = torch.long)

edge_index

# %%
x = torch.tensor([[0, 2], [-1, 2], [2, 2], [0, 0]], dtype = torch.float)
x

# %%
graph = Data(x = x, edge_index = edge_index.T.contiguous())
graph

# %%
print(
    graph.num_nodes,
    "\n",
    graph.num_edges,
    "\n",
    graph.num_node_features,
    "\n",
    graph.num_edge_features,
    "\n",
    graph.num_features
)

# %%
print(
    graph.contains_isolated_nodes(),
    "\n",
    graph.contains_self_loops(),
    "\n",
    graph.is_directed(),
    "\n",
    graph.is_undirected(),
    "\n"
)

# %%

#%%
import torch
from torch_geometric.datasets import KarateClub, GNNBenchmarkDataset, Amazon, SuiteSparseMatrixCollection

# %%
graphs = SuiteSparseMatrixCollection(root = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/PyTorch/Pygeometric/datasets", group = "LPnetlib", name = "lp_adlittle")

graphs

# %%
graph = graphs[0]

graph

# %%
graph.edge_index

#%%
from utilities.DrawGraph import Draw

# %%
Draw(graph)

# %%
graph.y

# %%

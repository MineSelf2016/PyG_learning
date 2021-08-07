#%%
import torch 
from torch_geometric.datasets import Planetoid


# %%
graphs = Planetoid(root = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/PyTorch/Pygeometric/datasets", name = "Cora")

print(len(graphs))

# %%
graph = graphs[0]

graph

# %%
graphs.num_classes

# %%
len(graphs)

# %%
graph.x[0]

# %%
graph.test_mask.sum()

# %%
graph.train_mask.sum()

# %%
graph.val_mask.sum()

# %%

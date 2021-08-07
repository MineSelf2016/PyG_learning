#%%
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import AMiner


# %%
dblp_ct1 = TUDataset(root = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/PyTorch/Pygeometric/datasets", name = "DBLP_ct1", use_node_attr = True)

dblp_ct1

#%%
graph = dblp_ct1[0]

#%%
graph


# %%
aminer = AMiner(root = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/PyTorch/Pygeometric/datasets")

aminer

# %%
type(aminer)

# %%
dir(aminer)

# %%
aminer[0]

# %%
edge_index = aminer[0].edge_index_dict[("paper", "written by", "author")]

from torch_geometric.data import Data 
graph = Data(edge_index = edge_index)
graph

# %%
graph.num_nodes

# %%
graph.num_edges

# %%

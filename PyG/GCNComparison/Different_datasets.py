#%%
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops

# %%
graphs = Planetoid(root = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/PyTorch/Pygeometric/datasets", name = "CiteSeer")

#%%
len(graphs)


#%%
graph = graphs[0]

num_features = graphs.num_features
num_class = graphs.num_classes

num_features, num_class

# %%

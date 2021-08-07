#%%
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

#%%
x = torch.tensor([
    [2, 1],
    [5, 6],
    [3, 7],
    [12, 0]
], dtype = torch.double)

edge_index = torch.tensor([
    [0, 1],
    [1, 0],
    [2, 1],
    [3, 2],
    [0, 3]
]).T

x, edge_index

# %%
graph = Data(x, edge_index)

graph

# %%
class MPNN(MessagePassing):
    def __init__(self, input_features, aggr='add', flow='source_to_target', node_dim=-2):
        super().__init__(aggr=aggr, flow=flow, node_dim=node_dim)
        self.linear = nn.Linear(input_features, 1)

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        x = self.linear(x)

        self.propagate(edge_index, x = x)
        return x 

    def message(self, x_i, x_j):
        print("source, x_j: {}".format(x_j))
        print("source, x_i: {}".format(x_i))
        return x_j

    def update(self, x_i, x_j):
        return x_j * 1.2



model = MPNN()

model 

# %%
output = model(graph)

# %%
"""
MPNN 中的message 函数所做的事情：
x_j, 邻居 / source 节点，[num_edges, num_features]
x_i, 中心 / target 节点，[num_edges, num_features]

注意，x_i, x_j 是一个整体的思想，每次处理的是一整张图，x_i, x_j对应的位置上的元素就是邻居节点；该函数并不是每次处理一个节点时就调用一次。

现在，通过这两个向量，结合相应的处理方式（如MLP 等），就可以完成节点状态的更新。

"""
x_j = x[edge_index[0]]
x_j

# %%
x_i = x[edge_index[1]]
x_i

# %%
x_i = F.softmax((x_j - x_i).pow(2) + x_j, dim = 1)
x_i

# %%


# %%

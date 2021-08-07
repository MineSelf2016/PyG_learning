#%%
import torch 
import torch_geometric
from torch_geometric.nn import MessagePassing

#%%
import networkx as nx 
from torch_geometric.utils.convert import to_networkx
from torch_geometric.datasets import KarateClub

def Draw(graph):
    G = to_networkx(graph)
    nx.draw(G)


# %%
"""
创建自定义的MessagePassing Network

1. 继承MessagePassing 基类
2. 重载下面几个函数：
    - __init__：指定aggr = "add", "mean", or "max"
    - message(x_i, x_j): 指定v_j 对于v_i 的信息如何结合
    - propagate(edge_index, size = None)：在何时进行传播，该方法的调用会出现在自定义类中的forward() 方法中
    - update(aggr_out, ...)：指定该怎样更新当前节点的特征，如果不重载该方法，那么就使用 aggr = "add", "mean", "max" 处理后的作为特征
 
"""

class SGCNLayer(MessagePassing):
    def __init__(self, input_features, output_features, aggr='add', flow='source_to_target', node_dim=-2):
        super().__init__(aggr=aggr, flow=flow, node_dim=node_dim)

        self.linear = torch.nn.Linear(input_features, output_features)

    def forward(self, x, edge_index):
        # x = self.linear(x)
        return self.propagate(edge_index, x = x)

    def message(self, x_i, x_j):
        print("x_i: {}, x_j: {}".format(x_i.shape, x_j.shape))
        return 1.2 * x_j


num_samples = 4
num_features = 2
num_classes = 2

x = torch.tensor([
    [2, 1],
    [5, 6],
    [3, 7],
    [12, 0]
], dtype = torch.float)

edge_index = torch.tensor([
    [0, 1],
    [1, 0],
    [2, 1],
    [3, 2],
    [0, 3]
]).T

y = torch.tensor([
    [0],
    [1],
    [0],
    [1]
])

graph = torch_geometric.data.Data(x = x, edge_index = edge_index, y = y)

graph 

#%%
model = SGCNLayer(num_features, num_classes)
print(model)

# %%
output = model(graph.x, graph.edge_index)
output

# %%
graph.x

# %%

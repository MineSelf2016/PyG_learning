#%%
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops

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
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.linear(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype = x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        return self.propagate(edge_index, x = x, norm = norm)


    def message(self, x_i, x_j, norm):
        # print("x_i: {}, x_j: {}".format(x_i.shape, x_j.shape))
        return norm.view(-1, 1) * x_j


# %%
class GCNModel(torch.nn.Module):
    def __init__(self, num_features, num_class):
        super(GCNModel, self).__init__()
        self.conv1 = SGCNLayer(num_features, 50)
        self.conv2 = SGCNLayer(50, num_class)
    
    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training = self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim = 1)
        return x 

# %%

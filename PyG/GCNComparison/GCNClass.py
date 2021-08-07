#%%
"""
1. 下载数据；
2. 处理数据，获取元数据；
3. 搭建网络，设置GCN 层
4. 训练网络

"""
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# %%
class GCNModel(torch.nn.Module):
    def __init__(self, num_features, num_class):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(num_features, 50)
        self.conv2 = GCNConv(50, num_class)
    
    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training = self.training)
        x = self.conv2(x, edge_index)
        x = F.log_softmax(x, dim = 1)
        return x 




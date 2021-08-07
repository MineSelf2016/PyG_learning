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

#%%
class GCNLayer(nn.Module):

    def __init__(self, input_features, output_features):
        super().__init__()
        self.linear = nn.Linear(input_features, output_features)

    def __call__(self, x, edge_index):
        """
        x, node_features, 要过滤的特征
        edge_index, 邻接关系，要融入到前向传播过程中

        """
        x = self.forward(x, edge_index)
        return x

    def forward(self, x, edge_index):
        adjacent_matrix = self.get_adjacent_matrix(x, edge_index)

        _A = self.get_A(adjacent_matrix)
        _D = self.get_D(adjacent_matrix)

        x = _D @ _A @ _D @ x
        
        x = self.linear(x)
        
        return x

    def get_adjacent_matrix(self, x, edge_index):
        num_nodes = x.shape[0]
        adjacent_matrix = torch.zeros(num_nodes, num_nodes)
        
        for edge in edge_index.T:
            adjacent_matrix[edge[0]][edge[1]] = 1

        return adjacent_matrix

    def get_A(self, adjacent_matrix):
        # return adjacent_matrix + torch.diag(torch.tensor([1 for _ in range(adjacent_matrix.shape[0])], dtype = torch.float))
        return adjacent_matrix + torch.eye(adjacent_matrix.shape[0])
    

    def get_D(self, adjacent_matrix):
        return torch.diag(self.get_A(adjacent_matrix).sum(dim = 0).pow(-1 / 2.))


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

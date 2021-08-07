#%%
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
# from torch_geometric.nn import GCNConv

#%%
epochs = 10
learning_rate = 0.1
graphs = Planetoid(root = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/PyTorch/Pygeometric/datasets", name = "Cora")

#%%
"""
one single 'large' graph

"""
graph = graphs[0]
graph

#%%
class GCNLayer(MessagePassing):
    def __init__(self, input_features, output_features, aggr='add', flow='source_to_target', node_dim=-2):
        super().__init__(aggr=aggr, flow=flow, node_dim=node_dim)
        self.linear = nn.Linear(input_features, output_features)


    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index)
        x = self.linear(x)

        row, col = edge_index
        deg = degree(col, x.size(0), dtype = x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]


        return self.propagate(edge_index, x = x, norm = norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

#%%
class GCNModel(nn.Module):
    def __init__(self):
        super(GCNModel, self).__init__()
        self.conv1 = GCNLayer(graphs.num_features, 16)
        self.conv2 = GCNLayer(16, graphs.num_classes)

    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.log_softmax(x, dim = 1)
        return x

#%%
model = GCNModel()
model 

# %%
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(epochs):
    output = model(graph)
    loss = loss_fn(output, graph.y)
    print("epoch: {}, loss: {}".format(epoch, loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# %%

#%%
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

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
class GCNModel(nn.Module):
    def __init__(self):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(graphs.num_features, 16)
        self.conv2 = GCNConv(16, graphs.num_classes)

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

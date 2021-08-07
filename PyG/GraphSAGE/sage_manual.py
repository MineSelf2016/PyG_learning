#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import networkx as nx 

import torch 
import torch.nn as nn 
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid

#%%
PubMed = Planetoid(root = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/PyTorch/Pygeometric/datasets", name = "PubMed")

#%%
graph = PubMed[0]

#%%
class SAGELayer(nn.Module):
    def __init__(self, in_features, out_features, aggr = "mean"):
        super(SAGELayer, self).__init__()
        self.in_features = in_features

        self.linear_1 = nn.Linear(in_features, out_features)
        self.linear_2 = nn.Linear(in_features, out_features)


    def forward(self, x, edge_index):
        x = self.linear_1(x) + self.linear_2(self.aggregate(x, edge_index))

        return x


    def aggregate(self, x, edge_index):
        adj_list = self.neighbor(x, edge_index)
        h = torch.ones(x.shape[0], self.in_features)
        
        for i in range(x.shape[0]):
            h[i] = x[adj_list[i]].mean(dim = 0).clone()

        return h

    def neighbor(self, x, edge_index):
        adj_list = {}
        for i in range(x.shape[0]):
            adj_list[i] = []

        E = edge_index.T
        for item in E:
            adj_list[item[0].item()].append(item[1].item())

        return adj_list        

#%%
class SAGEModel(nn.Module):

    def __init__(self):
        super(SAGEModel, self).__init__()
        self.sage_1 = SAGELayer(in_features = 500, out_features = 20)

        self.sage_2 = SAGELayer(in_features = 20, out_features = 3)

    def forward(self, x, edge_index):
        x = self.sage_1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = 0.5, training = self.training)
        
        x = self.sage_2(x, edge_index)
        
        return x

#%%
model = SAGEModel()
print(model)

# %%
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(graph.x, graph.edge_index)  # Perform a single forward pass.
      loss = criterion(out[graph.train_mask], graph.y[graph.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test():
      model.eval()
      out = model(graph.x, graph.edge_index)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[graph.test_mask] == graph.y[graph.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(graph.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc

#%%
for epoch in range(1, 21):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

#%%
test_acc = test()
print(f'Test Accuracy: {test_acc:.4f}')

#%%

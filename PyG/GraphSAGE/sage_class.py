#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn 
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv

#%%
PubMed = Planetoid(root = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/PyTorch/Pygeometric/datasets", name = "PubMed")

graph = PubMed[0]
graph

#%%
class SAGEModel(nn.Module):

    def __init__(self):
        super(SAGEModel, self).__init__()
        self.sage_1 = SAGEConv(in_channels = 500, out_channels = 20)
        self.sage_2 = SAGEConv(in_channels = 20, out_channels = 3)

    def forward(self, x, edge_index):
        x = self.sage_1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p = 0.5, training = self.training)
        
        x = self.sage_2(x, edge_index)
        # x = F.relu(x)
        # x = F.dropout(x, p = 0.5, training = self.training)

        # x = self.linear(x)
        
        return x

model = SAGEModel()
print(model)
# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
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

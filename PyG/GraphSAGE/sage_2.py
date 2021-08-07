#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import networkx as nx 

#%%
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch_geometric
from torch_geometric.datasets import Planetoid

#%%
Cora = Planetoid(root = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/PyTorch/Pygeometric/datasets", name = "Cora")

#%%
graph = Cora[0]

#%%
graph

#%%
graph.x.shape

#%%
graph

#%%
graph.edge_index

#%%
e = graph.edge_index.numpy().T.tolist()

#%%
class SAGELayer(nn.Module):
    def __init__(self, edge_index, depth, num_neighbors, in_features, out_features, aggr = "mean"):
        super(SAGELayer, self).__init__()

        self.G = nx.Graph()
        self.G.add_edges_from(edge_index)

        self.linear_1 = nn.Linear(2 * in_features, 20)
        self.linear_2 = nn.Linear(20 * 2, out_features)

        self.depth =  depth
        self.num_neighbors = num_neighbors
        self.aggr = aggr

    def forward(self, x):
        h_old = x
        h_new = None

        for d in range(self.depth):
            if d == 0:
                h_new = torch.zeros(2708, 20)
            elif d == 1:
                h_new = torch.zeros(2708, 10)
            else:
                raise Exception("depth out of range")

            for node_v in self.G.nodes:
                neighbors = self.sample(node_v)            
                h_neighbor = self.aggregate(neighbors, h_old)

                h_v = torch.cat([h_old[node_v], h_neighbor], dim = 0)

                if d == 0:
                    h_v = self.linear_1(h_v)
                elif d == 1:
                    h_v = self.linear_2(h_v)
                else:
                    raise Exception("depth out of range")

                h_v = F.sigmoid(h_v)
                # print(h_neighbor[h_neighbor != 0])
                h_v = h_v / torch.norm(h_v, p = 2)

                h_new[node_v] = h_v
            
            h_old = h_new

        return h_new
        

    def sample(self, node_v):
        neighbors = list(self.G.adj[node_v])
        
        if len(neighbors) >= self.num_neighbors:
            neighbors = np.random.choice(neighbors, size = self.num_neighbors, replace = False)
        else:
            neighbors = np.random.choice(neighbors, size = self.num_neighbors, replace = True)
        
        return neighbors

    def aggregate(self, neighbors, x):
        if self.aggr == "mean":
            idx, value = x[neighbors].max(dim = 0)
            return value

        else:
            print("not mean aggregate")
            return None
        

#%%
class GraphSAGE(nn.Module):
    def __init__(self, edge_index, depth, num_neighbors, in_features, out_features, aggr = "mean"):
        super(GraphSAGE, self).__init__()
        self.Sage_1 = SAGELayer(edge_index, depth = 2, num_neighbors = 4, in_features = 1433, out_features = 10)
        self.linear = nn.Linear(10, 7)
    
    def forward(self, x):
        x = self.Sage_1(x)
        x = x.relu()
        x = F.dropout(x, p = 0.5, training = self.training)
        x = self.linear(x)

        return x 

model = GraphSAGE(e, depth = 2, num_neighbors = 4, in_features = 1433, out_features = 10)
#%%
print(model)

#%%
optimizer = torch.optim.Adam(model.parameters(), lr = 0.2, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

#%%
def train():
      model.train()
      optimizer.zero_grad()  # Clear gradients.

      output = model(graph.x)  # Perform a single forward pass.
      loss = criterion(output[graph.train_mask], graph.y[graph.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test():
      model.eval()
      output = model(graph.x)
      pred = output.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[graph.test_mask] == graph.y[graph.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(graph.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc

#%%
for epoch in range(1, 201):
    loss = train()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

    if epoch % 20 == 0:
        test_acc = test()
        print(f'Test Accuracy: {test_acc:.4f}')

#%%


#%%
graph.x


#%%
for node in model.G.nodes:
    print(node)


# #%%
# class GCNModel(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(GCNModel, self).__init__()
#         self.gcn_1 = nn.Linear(in_features, out_features)
    
#     def forward(self, graph, x):
#         x = self.gcn_1(x)

#         return x
        

# model = GCNModel(10, 2)
# #%%
# output = model(None, torch.ones(10))

# output
# #%%


# %%

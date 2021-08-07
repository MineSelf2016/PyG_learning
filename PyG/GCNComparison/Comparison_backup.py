#%%
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree, add_self_loops

#%%
import torch
import random
import numpy as np 
import os

seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)




# %%
graphs = Planetoid(root = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/PyTorch/Pygeometric/datasets", name = "Cora")

graph = graphs[0]

num_features = graphs.num_features
num_class = graphs.num_classes

num_features, num_class


#%%
import GCNClass
import GCNMannual
import MPNN
models = {
    "GCNClass": GCNClass.GCNModel(num_features, num_class),
    "GCNMannual": GCNMannual.GCNModel(num_features, num_class),
    "MPNN": MPNN.GCNModel(num_features, num_class)

}

#%%
model = models["MPNN"]
print(model)

# %%
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)

# %%
model.train()
for epoch in range(10):
    
    output = model(graph)

    loss = loss_fn(output[graph.train_mask], graph.y[graph.train_mask])

    print("epoch: {}, loss: {}".format(epoch, loss.item()))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# %%
model.eval()
output = model(graph)
_, y_pred = output.max(dim = 1)
correct_num = (y_pred[graph.test_mask] == graph.y[graph.test_mask]).sum().item()
total_num = graph.test_mask.sum().item()

#%%
print("correct rate: {:.4f}".format(correct_num / total_num))

# %%

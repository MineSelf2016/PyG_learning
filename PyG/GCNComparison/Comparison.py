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
import GCNClass
import GCNMannual
import MPNN

#%%
import torch
import random
import numpy as np 
import os

seed = 8808
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)


# %%
Cora = Planetoid(root = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/PyTorch/Pygeometric/datasets", name = "Cora")

CiteSeer = Planetoid(root = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/PyTorch/Pygeometric/datasets", name = "CiteSeer")

PubMed = Planetoid(root = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/PyTorch/Pygeometric/datasets", name = "PubMed")


graphs = {
    "Cora": Cora,
    "CiteSeer": CiteSeer,
    "PubMed": PubMed
}

def cal_correct_rate(model, graph):
    model.eval()
    output = model(graph)
    _, y_pred = output.max(dim = 1)
    
    correct_num_test = (y_pred[graph.test_mask] == graph.y[graph.test_mask]).sum().item()
    total_num_test = graph.test_mask.sum().item()

    correct_num_train = (y_pred[graph.train_mask] == graph.y[graph.train_mask]).sum().item()
    total_num_train = graph.train_mask.sum().item()


    print("\ttestset correct rate: {:.6f}".format(correct_num_test / total_num_test))

    print("\ttrainset correct rate: {:.6f}".format(correct_num_train / total_num_train))

    print()
    
    model.train()

#%%
def test_model(model, graph):
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)

    model.train()
    for epoch in range(501):
        
        output = model(graph)

        # loss = loss_fn(output[graph.train_mask], graph.y[graph.train_mask])
        loss = loss_fn(output, graph.y)

        if epoch % 100 == 0:
            print("epoch: {}, loss: {}".format(epoch, loss.item()))            
            cal_correct_rate(model, graph)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


# %%
def test_graph(graph_list):
    num_features = graph_list.num_features
    num_class = graph_list.num_classes
    models = {
        "GCNClass": GCNClass.GCNModel(num_features, num_class),
        "GCNMannual": GCNMannual.GCNModel(num_features, num_class),
        "MPNN": MPNN.GCNModel(num_features, num_class)
    }

    graph = graph_list[0]
    for name, model in models.items():
        print(name)
        test_model(model, graph)

#%%
for name, graph in graphs.items():
    print("graph name {}".format(name))
    test_graph(graph)
    print("\n\n")


#%%

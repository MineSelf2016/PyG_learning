#%%
import torch 
from torch_geometric.data import Data 
from torch_geometric.datasets import Planetoid

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


#%%
"""
First, we need to load the Cora dataset from Planetoid

"""
graphs = Planetoid(root = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/PyTorch/Pygeometric/datasets", name = "Cora")

# %%
graph = graphs[0]

graph

# %%
"""
Now, we have the data, and the data is in a graph style, so we can just implement a GNN layer then learn the parameters on the graph.

"""
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCNModel(torch.nn.Module):

    """
    定义两个GCN 层，这两个层的输入参数是在graphs 上进行的操作
    GCN 层的源码实现是怎样的，需要学一下

    注意，这里的 __init__ 方法中，并没有存储任何的数据集数据，只有两层的模型参数，那么，训练所需要的数据都是在forward 方法执行的时候输入进来的
    """
    def __init__(self):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(graphs.num_node_features, 16)
        self.conv2 = GCNConv(16, graphs.num_classes)

    """
    forward 函数是重中之重，一定要理解这里是如何前向传播的：
    forward 的参数是graph，所以，每次进行forward 时，哪怕是不同的graph 结构，GCN 层也是能够处理的，只要 graph.num_features 是相同的就可以。

    那么，GCNConv 就是最关键的一个类，这个类的对象的 __cal__ 产生的效果就是完成一次 filter。
    """
    def forward(self, graph):
        x, edge_index = graph.x, graph.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training = self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim = 1)

# %%
"""
初始化一个 GCN model，然后设置优化器，损失函数。

"""
model = GCNModel()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 5e-4)
loss_fn = torch.nn.NLLLoss()

#%%
model.train()
for epoch in range(200):
    output = model(graph)
    loss = loss_fn(output[graph.train_mask], graph.y[graph.train_mask])
    # print("epoch: {}, loss: {}".format(epoch, loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


#%%
model.eval()
_, pred = model(graph).max(dim = 1)
num_correct = (pred[graph.test_mask] == graph.y[graph.test_mask]).sum().item()

print("correct rate: {}".format(num_correct / graph.test_mask.sum().item()))

# %%


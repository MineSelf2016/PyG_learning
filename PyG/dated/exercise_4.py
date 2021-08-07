#%%
"""
Batching along new dimensions 

this is typically used in the graph-level label, which will return the graph feature vector while using batch operation.

inherit the Data class and overwrite __cat_dim__(self, key, value) method


"""



#%%
import torch
from torch_geometric.data import Data 
from torch_geometric.data import DataLoader

#%%
class MyData(Data):
    def __cat_dim__(self, key, value):
        if key == "foo":
            return 0
        else:
            return super().__cat_dim__(key, value)

#%%
foo_1 = torch.randn(10)
foo_2 = torch.randn(10)


graph_1 = MyData(
    x = torch.tensor([[3, 4], [-1, 2], [-5, 8]], dtype = torch.float),
    edge_index = torch.tensor([
        [0, 1],
        [1, 0],
        [1, 2],
        [2, 1]
    ]).T ,
    foo = foo_1
)

graph_2 = MyData(
    x = torch.tensor([[-5, -10], [2.5, 3], [8, 9]], dtype = torch.float),
    edge_index = torch.tensor([
        [0, 1],
        [1, 0],
        [1, 2]
    ]).T ,
    foo = foo_2
)

graph_1, graph_2

# %%
loader = DataLoader([graph_1, graph_2], batch_size = 2)

batch = next(iter(loader))
batch

#%%

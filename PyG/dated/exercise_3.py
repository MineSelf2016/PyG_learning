#%%
import torch 
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.data import DataLoader

root_path = "/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/PyTorch/Pygeometric/datasets"

#%%
graphs = Planetoid(root = root_path, name = "Cora")
graphs 

# %%
graphs.num_classes

# %%
graphs.num_node_features

# %%
graphs.len()

# %%
graphs.shuffle()

# %%
graphs = TUDataset(root = root_path, name = "ENZYMES", use_node_attr = True)
loader = DataLoader(graphs, batch_size = 32, shuffle = False)

# %%
from collections import Counter 
for i, batch in enumerate(loader):
    if i == 0:
        # print()
        # cc = Counter(batch.batch.numpy().tolist())
        # print(cc)
        # sorted(cc.items(), key = lambda kv : kv[1])
        # print(len(batch))
        # print(batch.batch)
        # print(batch)
        print(batch.x.shape)
        print(batch.batch)
        # for g in batch:
        #     print(g)
    else:
        break 

# %%
graphs[0]

# %%
graphs[1]

# %%
graphs[2]

# %%
graphs[3]

# %%

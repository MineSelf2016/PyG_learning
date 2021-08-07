#%%
import torch 
from torch_geometric.data import Data 
import torch_geometric.transforms as T 

# %%
edge_index = torch.tensor([
    [0, 1],
    [1, 0],
    [1, 2],
    [2, 1]
])

x = torch.tensor([
    [0, 0.1],
    [2, -3.2],
    [0, 5]
], dtype = torch.float)

graph = Data(edge_index = edge_index.T, x = x)

# %%
graph

# %%
"""
ToSparse method


"""
def test_sparse(graph):
    trans = T.ToSparseTensor()

    graph = trans(graph)

    return graph

# print(test_sparse(graph))

# %%
def test_dense():
    dense = T.ToDense(num_nodes = 2)

    i = torch.tensor([[0, 1, 1], [2, 0, 2]])
    v = torch.tensor([0, 1, 2])
    sparse_edge_index = torch.sparse_coo_tensor(i, v, [2, 3])

    graph = Data(edge_index = sparse_edge_index)
    graph = dense(graph)

    graph

# %%
def test_constant():
    constant = T.Constant(value = 10)

    output = constant(graph)

    return output.x

# %%
graph.edge_index

# %%
def test_onehot():
    onehot = T.OneHotDegree(max_degree = 3)
    output = onehot(graph)

    return output

# %%
def test_self_loops():
    self_loops = T.AddSelfLoops()
    self_loops(graph)

    return graph 

graph.edge_index


# %%
def test_two_hop():
    edge_index = torch.tensor([
        [0, 1],
        [1, 2],
    ])

    graph.edge_index = edge_index

    graph.edge_index

    twohop = T.TwoHop()

    twohop(graph)

    return graph

# %%
def test_undirected():
    edge_index = torch.tensor([
            [0, 1],
            [1, 2],
        ])

    graph.edge_index = edge_index

    undirected = T.ToUndirected()

    undirected(graph)

    return graph

# %%

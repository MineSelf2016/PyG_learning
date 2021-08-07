#%%
import networkx as nx 
from torch_geometric.utils.convert import to_networkx
from torch_geometric.datasets import KarateClub

def Draw(graph):
    G = to_networkx(graph)
    nx.draw(G)


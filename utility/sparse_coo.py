#%%
import torch 

# %%
indices = [[0, 2], [1, 0], [1, 2]]
values = [3, 4, 5]

indices, values 

# %%
x = torch.sparse_coo_tensor(torch.tensor(indices).T, values, size = (2, 3))

x

# %%
x.to_dense()

# %%
torch.sparse_coo_tensor(size = (3, 5)).to_dense()

# %%
indices = [[1, 1], [1, 2]]
values = [[3, 4], [4, 5]]

x = torch.sparse_coo_tensor(indices = torch.tensor(indices).T, values = values, size = (2, 3, 2))
x

#%%
x.is_coalesced()

# %%
x.to_dense()

# %%
x.layout == torch.sparse_coo

# %%
x.is_sparse

# %%
x.layout == torch.sparse_coo

# %%
x

#%%
x.sparse_dim()


#%%
x.dense_dim()

#%%
x.to_dense()

#%%
x = x.coalesce()

#%%
x.is_coalesced()


#%%
x.indices()

#%%
x

#%%
x.values()


# %%
y = torch.randn(4)
y

# %%
y.layout

# %%
indices = [[1, 1], [1, 2]]
values = [[3, 4], [4, 5]]

x = torch.sparse_coo_tensor(indices = torch.tensor(indices).T, values = values, size = (2, 3, 2))
x

# %%
x.to_dense()

# %%
x[0, 0, 1:]

# %%
x = torch.randn(2, 3).to_sparse().requires_grad_(True)
x

# %%
y = torch.randn(3, 2, requires_grad = True)
y

# %%
output = torch.sparse.mm(x, y)
output

#%%
output.sum().backward()

# %%
x.grad

# %%
y.grad.data

# %%

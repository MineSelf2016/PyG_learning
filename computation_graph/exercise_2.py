#%%
import torch

# %%
x = torch.empty(1, 100000)
x

#%%
x = x.bernoulli(p = 0.5)
x

#%%
x.sum()

#%%
torch.bitwise_not(torch.tensor([False, False, True], dtype=torch.bool))


# %%
x = torch.tensor([1, 2, 3])
y = x.expand((10, 3))
y

# %%
x[0] = 8

# %%
y
# %%
torch.mm(y, torch.tensor([[1, 0, 0]]).T)

# %%
x = torch.tensor([0, 1, 1.5, -0.5])
torch.ceil(x)

# %%
x = torch.ones([4, 20])
x 

# %%
x_split = x.chunk(10, dim = 1)
for i, element in enumerate(x_split):
    print("epoch: ", i, ", element: ", element)
    print()


# %%
x = torch.randn((1, 10))
x

# %%
x.clamp(min = -0.2, max = 0.5)

# %%
x = torch.randn((1, 10))
x

#%%
x.data_ptr()

# %%
x.logcumsumexp(dim = 0)

# %%
x = torch.randn(4)
x

# %%
y = x.diag()
y

# %%
torch.diagflat(x)


# %%

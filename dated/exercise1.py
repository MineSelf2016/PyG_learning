#%%
import torch

a = torch.tensor([2, 3], requires_grad = True, dtype = torch.float)
b = torch.tensor([6, 4], requires_grad = True, dtype = torch.float)

Q = 3 * a ** 3 - b ** 2

type(Q)

#%%
external_grad = torch.tensor([1., 1.])
Q.backward(gradient = external_grad)


# %%
a.grad, b.grad, Q.grad


"""
DAGs are dynamic in PyTorch An important thing to note is that 
the graph is recreated from scratch; 
after each .backward() call, autograd starts populating a new graph. 
This is exactly what allows you to use control flow statements in your model; 
you can change the shape, size and operations at every iteration if needed.

"""

# %%
Q.grad_fn(b)


# %%

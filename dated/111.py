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
input = torch.randn([100, 4])
target = torch.randn([1, 100])

w = torch.randn([4, 1], requires_grad = True)

#%%
# print("first w is leaf: ", w.is_leaf, " w's grad: ", w.requires_grad)
learning_rate = 0.2
for i in range(10):
    # loss = forward(w, input)
    output = torch.mm(w.T, input.T)
    output = torch.nn.functional.leaky_relu(output)
    # output = torch.nn.functional.sigmoid(output)
    # output = torch.nn.functional.tanh(output)
    loss = (((target - output).abs().pow(3)) / 100).sum()
    print("epoch: ", i, " loss: ", loss)

    loss.backward(torch.ones_like(loss), retain_graph = True, create_graph = True)
    
    w.data = w.data - learning_rate * w.grad.data
    # print("second w is leaf: ", w.is_leaf, " w's grad: ", w.requires_grad)
    # print(w.grad.data)
    print()
    w.grad.data.zero_()

#%%
w.grad.data

#%%
"""
Default gradient layouts
When a non-sparse param receives a non-sparse gradient during torch.autograd.backward() or torch.Tensor.backward() param.grad is accumulated as follows.

If param.grad is initially None:

If param’s memory is non-overlapping and dense, .grad is created with strides matching param (thus matching param’s layout).

Otherwise, .grad is created with rowmajor-contiguous strides.

If param already has a non-sparse .grad attribute:

If create_graph=False, backward() accumulates into .grad in-place, which preserves its strides.

If create_graph=True, backward() replaces .grad with a new tensor .grad + new grad, which attempts (but does not guarantee) matching the preexisting .grad’s strides.

The default behavior (letting .grads be None before the first backward(), such that their layout is created according to 1 or 2, and retained over time according to 3 or 4) is recommended for best performance. Calls to model.zero_grad() or optimizer.zero_grad() will not affect .grad layouts.


"""


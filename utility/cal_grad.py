#%%
import torch 


#%%
x = torch.tensor([2], requires_grad = True, dtype = torch.double)
# x.requires_grad = True
w = torch.Tensor([8])
w.requires_grad = True

x, w 
#%%
a = w + x
b = w + 1
output = b / a 

#%%
a.grad_fn, b.grad_fn, output.grad_fn \
    ,x.grad_fn, w.grad_fn

#%%
output.backward()

#%%
print(w.data)
print(w.grad.data)


# %%
"""
使用backward()函数反向传播计算tensor的梯度时，
并不计算所有tensor的梯度，
而是只计算满足这几个条件的tensor的梯度：
1.类型为叶子节点 且 requires_grad=True的tensor；
2.依赖该叶子节点 且 requires_grad=True的tensor。

"""


# %%

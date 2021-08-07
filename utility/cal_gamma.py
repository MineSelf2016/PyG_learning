#%%
"""
calculate the gamma function

https://zhuanlan.zhihu.com/p/147583667

https://pytorch.org/docs/stable/generated/torch.lgamma.html#torch.lgamma

"""

#%%
import torch

#%%
exponent = 3
x_data = torch.arange(0, 20, step = 0.1)
x = x_data
# x

# %%
y_data = torch.pow(x, exponent) / torch.exp(x)
# y_data

# %%
import matplotlib.pyplot as plt 
plt.plot(x_data, y_data)

# %%
"""
use MCMC to calculate the integral 

"""
def get_height(x, exponent = 3):
    x = torch.tensor(x, dtype = torch.double)
    return (torch.pow(x, exponent) / torch.exp(x)).item()


#%%
get_height(3)

#%%
width = 20
import random
ss = 0
num_itr = 10000

for i in range(num_itr):
    x = random.uniform(0, width)
    y_height = get_height(x)
    s = width * y_height
    ss += s

area = ss / num_itr
area

# %%

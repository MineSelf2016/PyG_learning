#%%
import torch

#%%
"""
Generally, in the classification tasks, we get the y_label as their category, such as y_label = [0, 1, 2, 3, 3, 2, ...], and we will convert it into the one-hot code. There are several methods can do this:

1. traditional for loop;

2. torch_scatter method.

"""


#%%
"""
Generate random y_label

"""

#%%
import numpy as np 

y_label = torch.from_numpy(np.random.randint(0, 4, 10))
y_label



#%%
def traditional_for_loop(y_label):
    y = torch.zeros(10, 4)
    for i, label in enumerate(y_label):
        y[i][label] = 1
    print(y)

traditional_for_loop(y_label)

#%%
def torch_scatter(y_label):
    y = torch.zeros(10, 4)
    y.scatter_(1, torch.tensor(y_label).unsqueeze(1), 1)
    print(y)


torch_scatter(y_label)

#%%

#%%
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import numpy as np 

# %%
x  = torch.randn(100, 4)
x

# %%
target = torch.randn(1, 100)
target

# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        return x 

net = Net()

#%%
output = net(x)
output.size()

#%%
for i, param in enumerate(net.parameters()):
    print(i, param)

#%%
param

#%%
dir(param)


#%%
criterion = nn.MSELoss()
loss = criterion(output, target)
loss.backward()



# %%
# for param in net.parameters():
#     print(type(param), param.size())
    # print(param)

# %%

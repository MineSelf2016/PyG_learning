#%%
import torch, torchvision
from torch import nn, optim 

#%%
model = torchvision.models.resnet18(pretrained = True)

for param in model.parameters():
    param.requires_grad = False

# %%
model.fc = nn.Linear(512, 10)

optimizer = optim.SGD(model.fc.parameters(), lr = 0.01, momentum = 0.9)


# %%

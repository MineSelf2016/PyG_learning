#%%
import torch, torchvision

model = torchvision.models.resnet18(pretrained = True)
data = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

#%%
prediction = model(data)


# %%
loss = (prediction - labels).sum()


# %%
loss.backward()

# %%
optim = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
optim.step()

# %%
prediction = model(data)
loss = (prediction - labels).sum()
loss.backward()
optim = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
optim.step()

# %%
i = 0
while i < 10:
    prediction = model(data)
    loss = (prediction - labels).sum()
    loss.backward()
    print(loss)
    optim = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.9)
    optim.step()
    i += 1

# %%

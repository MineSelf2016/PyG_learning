#%%
import torch
import torch.nn as nn 
import torch.nn.functional as F 
# from torch_geometric.datasets import Planetoid

# %%
num_samples = 100
num_features = 4
num_classes = 4
epochs = 10
learning_rate = 0.1

#%%
x = torch.randn(num_samples, num_features)
target = torch.randn(num_samples, num_classes)

#%%
_, indices = target.max(dim = 1)
target = F.one_hot(indices, num_classes = num_classes)
target.shape

#%%
class FFNN(nn.Module):
    def __init__(self):
        super(FFNN, self).__init__()
        self.layer1 = nn.Linear(num_features, 10)
        self.layer2 = nn.Linear(10, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        # x = F.log_softmax(x, dim = 1)
        return x

#%%
model = FFNN()
model 

# %%
loss_fn = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

for epoch in range(epochs):
    output = model(x)
    print(output.shape)

    loss = loss_fn(output, indices)
    print("epoch: {}, loss: {}".format(epoch, loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# %%

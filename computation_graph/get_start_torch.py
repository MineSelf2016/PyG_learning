#%%
import torch
import torch.nn as nn 
import torch.nn.functional as F 

#%%
num_samples = 100
num_features = 1
input = torch.randn(num_samples, num_features)
target = torch.randn(num_samples, 1)

#%%
class Net(nn.Module):
    def __init__(self, input_features, output_features):
        super(Net, self).__init__()
        self.linear = nn.Linear(input_features, output_features)

    def forward(self, x):
        x = self.linear(x)
        x = F.relu(x)
        return x 


net = Net(num_features, 1)
print(net)

#%%
def SGD(net, learning_rate = 0.1):
    for f in net.parameters():
        f.data.sub_(f.grad.data * learning_rate)


#%%
criterion = nn.MSELoss()

#%%
for i in range(10):
    out = net(input)
    loss = criterion(out, target)
    loss.backward()
    print(loss.item())
    SGD(net)
    net.zero_grad()

# %%

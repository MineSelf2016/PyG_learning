# %%
import torch
import torch.nn as nn 
import torch.nn.functional as F 

# %%
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 3)
        self.fc2 = nn.Linear(3, 5)
        self.fc3 = nn.Linear(5, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# %%
net = Net()
print(net)

# %%
input = torch.randn(100, 4)
output = net(input)

# %%
print(output.shape)


# %%
target = torch.ones(100, dtype = torch.long)

#%%
criterion = nn.CrossEntropyLoss()

loss = criterion(output, target)
print(loss)

# %%
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr = 0.01)

# %%
i = 0
while i < 100:
    optimizer.zero_grad()

    output = net(input)
    loss = criterion(output, target)
    print(loss)

    loss.backward()
    optimizer.step()

    i += 1

# %%
def loss_function(x, label):
    return -x[label] + torch.log(torch.sum(torch.exp(x)))

# %%
ss = 0
for i in range(100):
    out = net.forward(input[i])
    ss += loss_function(out, label = 1)

ss /= 100
print(ss)

# %%
output = net(input)
loss = criterion(output, target)
print(loss)

# %%
net.state_dict()

#%%
torch.save(net.state_dict(), "state.pkl")
net_state_dict = torch.load("state.pkl")
new_net = Net()
new_net.load_state_dict(net_state_dict)

# %%
new_net.state_dict()

# %%

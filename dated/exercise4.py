# %%
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

# %%
class Net(nn.Module):

    """
    torch.nn only supports mini-batches. 
    The entire torch.nn package only supports inputs that are 
    a mini-batch of samples, and not a single sample.

    For example, nn.Conv2d will take in a 4D Tensor of 

    nSamples x nChannels x Height x Width

    """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    """
    You just have to define the forward function, and 
    the backward function (where gradients are computed) 
    is automatically defined for you using autograd. 
    You can use any of the Tensor operations in the forward function.

    """

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1: ]
        print("type of size: ", type(size))
        print("size: ", size)

        num_features = 1
        for s in size:
            num_features *= s 
        return num_features

# %%
net = Net()
print(net)

# %%
params = list(net.parameters())
print(len(params))
print(params[0].size())
print(params[1].size())

# %%
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# %%
net.zero_grad()
out.backward(torch.randn(1, 10))

# %%
output = net(input)
target = torch.randn(10)
target = target.view(1, -1)

criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)

# %%
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# %%
# We can implement SGD using simple Python code:
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)


# %%
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr = 0.01)


# in your training loop:
"""
You need to clear the existing gradients though, 
else gradients will be accumulated to existing gradients.
"""
i = 0
while i < 10:
    optimizer.zero_grad()
    
    output = net(input)
    loss = criterion(output, target)
    print(loss)
    loss.backward()
    optimizer.step()

    i += 1

# %%

#%%
import torch
from torch import nn 

# %%
class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 4),
            nn.ReLU()
        )

    def forward(self, x):
        output = self.linear_relu_stack(x)
        return output

# %%
x = torch.randn(10, 4)
y_label = [0, 1, 2, 3, 3, 2, 1, 0, 1, 2]
y = torch.zeros(10, 4)
y.scatter_(1, torch.tensor(y_label).unsqueeze(1), 1)
net = NeuralNetwork()
output = net(x)

# %%
output.size()

# %%

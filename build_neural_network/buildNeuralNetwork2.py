#%%
import torch
from torch import nn 
import numpy as np 

# %%
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.Linear_Relu_Stack = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 10),
            nn.ReLU()
        )

    def forward(self, x):
        output = self.Linear_Relu_Stack(x)
        return output

    def predict(self, x):
        y_pred = self.softmax(x).argmax(dim = 1)
        return y_pred

net = NeuralNetwork()

#%%
print("Model architechture: ", net, "\n\n")

for name, parameter in net.named_parameters():
    print(f"Layer: {name} | Size: {parameter.size()} | Parameter: {parameter[:2]}")

# %%
num_samples = 100
num_features = 4
num_categories = 10

x = torch.randn(num_samples, num_features)
y = np.random.randint(0, num_categories, num_samples)
y = torch.from_numpy(y)
y = torch.zeros(num_samples, num_categories).scatter_(dim = 1, index = y.unsqueeze(1), value = 1)

x.shape, y.shape

# %%
output = net(x)
output.shape

#%%
pred = net.predict(output)
pred

# %%

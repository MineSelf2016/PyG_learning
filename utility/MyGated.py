#%%
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

#%%
class MyDecisionGate(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if x.sum() > 0:
            # print("decision > 0")
            return x 
        else:
            # print("decision < 0")
            return -x


class MyCell(nn.Module):
    def __init__(self):
        super().__init__()
        self.dg = MyDecisionGate()
        self.linear = nn.Linear(4, 1)

    def forward(self, x, h):
        x = self.dg(x)
        x = F.tanh(x)
        x = x + h
        h = F.sigmoid(x)
        x = self.linear(x)
        return x, h

model = MyCell()
x = torch.randn(10, 4)
h = torch.randn(10, 4)
target = torch.randn(10, 1)

#%%
loss_fn = nn.MSELoss()

optimizer = torch.optim.Adam(model.parameters(), lr = 0.1)

# %%
for epoch in range(10):
    out_x, out_h = model(x, h)
    loss = loss_fn(out_x, target)
    print("epoch: {}, loss: {}".format(epoch, loss.item()))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# %%
traced_cell = torch.jit.trace(model, (x, h))
traced_cell

# %%
traced_cell.code

# %%
print(model(x, h))
print(traced_cell(x, h))

# %%
torch.allclose(model(x, h)[1], traced_cell(x, h)[1])

# %%

#%%
import os 
import torch

os.chdir("/Users/mineself2016/study/codeSpace/VSCodeWorkSpace/PythonWorkSpace/PyTorch")

torch.manual_seed(0)

# %%
x = torch.randn(100, 4)
w = torch.randn(4, 1, requires_grad = True)

y_target = torch.randn(1, 100)

#%%
def get_loss(y_pred, y_target) -> torch.Tensor:
    err = y_pred - y_target
    return (torch.pow(err, 2) / 100).sum()

# %%
learning_rate = 0.1

#%%
for i in range(10):
    y_pred = torch.mm(w.T, x.T)
    # y_pred = torch.nn.functional.relu(y_pred)
    loss = get_loss(y_pred, y_target)
    print("epoch: ", i, " loss: ", loss)
    print()
    loss.backward()
    w.data = w.data - learning_rate * w.grad.data
    w.grad.data = torch.zeros_like(w.grad.data)
    # w.grad.data.zero_()

#%%

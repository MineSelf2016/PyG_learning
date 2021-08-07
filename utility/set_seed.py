#%%
import torch
import random
import numpy as np 
import os

seed = 10
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

#%%
GLOBAL_SEED = 1
 
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
 
GLOBAL_WORKER_ID = None
def worker_init_fn(worker_id):
    global GLOBAL_WORKER_ID
    GLOBAL_WORKER_ID = worker_id
    set_seed(GLOBAL_SEED + worker_id)
 
# dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, worker_init_fn=worker_init_fn)


#%%
input = torch.randn([100, 4])
target = torch.randn([1, 100])

w = torch.randn([4, 1], requires_grad = True)

#%%
def forward(w, input):
    output = torch.mm(w.T, input.T)
    loss = ((target - output).pow(2) / 100).sum()
    return loss

#%%
# print("first w is leaf: ", w.is_leaf, " w's grad: ", w.requires_grad)
learning_rate = 0.2
for i in range(10):
    # loss = forward(w, input)
    output = torch.mm(w.T, input.T)
    output = torch.nn.functional.leaky_relu(output)
    # output = torch.nn.functional.sigmoid(output)
    # output = torch.nn.functional.tanh(output)
    loss = (((target - output).abs().pow(3)) / 100).sum()
    print("epoch: ", i, " loss: ", loss)

    loss.backward()
    w.data = w.data - learning_rate * w.grad.data
    # print("second w is leaf: ", w.is_leaf, " w's grad: ", w.requires_grad)
    print()
    w.grad.data.zero_()


# %%
# x = torch.tensor([1, 2, 3], requires_grad = True, dtype = torch.double)
# x

#%%
import torch
from torch.utils.data import Dataset, DataLoader

# %%
"""
dataloader can help us shuffle the training data after every iter, and yield training data in a minibatches at one time.

dataset can yield one sample with its training data and label respectively, and datasets can be implemented customly, just inherit the base class torch.utils.data.Dataset.

the custom datasets must override the following three functions:

1. __init__: initial the inner data variable and store some metadata;
2. __len__: the number of training data in our dataset;
3. __getitem__(self, index): yield a sample along with its data and label at one time.

4. __iter__(self): for IterableDataset. https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset


"""

#%%
class CustomDataset(Dataset):
    def __init__(self, data, label, transform = None, target_transform = None):
        super().__init__()
        self.data = data
        self.label = label


    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        data = self.data[index, :]
        label = self.label[index]

        return data, label


# %%
x = torch.randn(100, 4)
y = torch.randn(100)

# %%
traning_data = CustomDataset(data = x, label = y)

# %%
train_loader = DataLoader(dataset = traning_data, batch_size = 10, shuffle = True)

# %%


# %%
# w = torch.randn(4, 1, requires_grad = True)

# for epoch in range(10):
#     itr = iter(train_loader)
#     epoch_loss = 0
#     for i in range(100 // 10):
#         train_features, train_labels = next(itr)
#         out = torch.mm(w.T, train_features.T)
#         loss = torch.pow((out - train_labels), 2).sum() / 10
#         epoch_loss += loss.item()
#         loss.backward()
#         w.data = w.data - 0.01 * w.grad.data
#         w.grad.zero_()

#     print("epoch: ", epoch, " loss: ", epoch_loss)

#%%
w = torch.randn(4, 1, requires_grad = True)

for epoch in range(10):
    itr = iter(train_loader)
    for i in range(100 // 10):
        train_features, train_labels = next(itr)
        out = torch.mm(w.T, train_features.T)
        loss = torch.pow((out - train_labels), 2).sum() / 10
        loss.backward()
        w.data = w.data - 0.01 * w.grad.data
        w.grad.zero_()

    epoch_loss = torch.pow((torch.mm(w.T, x.T) - y), 2).sum() / 100
    print("epoch: ", epoch, " loss: ", epoch_loss.item())


# %%
train_features, train_labels = next(itr)
print(train_features)
print(train_labels)
print("Feature batch shape: ", train_features.size())
print("Label batch shape: ", train_labels.size())


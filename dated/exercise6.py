# %%
import torch
import torch.nn as nn 

# %%
def fizz_buzz_encode(i):
    if i % 15 == 0:
        return 3
    elif i % 5 == 0:
        return 2
    elif i % 3 == 0:
        return 1
    else:
        return 0

def fizz_buzz_decode(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

def helper(i):
    return fizz_buzz_decode(i, prediction = fizz_buzz_encode(i))

# %%
for i in range(1, 20):
    print(helper(i))

# %%
def binary_encoder(i):
    return torch.tensor([i >> d & 1 for d in range(32)][: : -1])


# %%
print(binary_encoder(100))

# %%

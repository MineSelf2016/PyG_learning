#%%
import torch 

# %%
x = torch.tensor([-1.5393, -0.8675,  0.5916,  1.6321])
y = torch.tensor([ 0.0967, -1.0511,  0.6295,  0.8360])


# %%
(x - y).abs().sum()


# %%
x.dist(y, p = 1)

# %%
x = torch.randn(4)
x

# %%
y = torch.randn(4)
y

# %%
x / y

# %%
torch.div(x, y, rounding_mode = "trunc")


# %%
torch.dot(torch.tensor([[2], [3]]), torch.tensor([[2, 1]]))

# %%
x = torch.tensor([[1, 2]])
x

# %%
x.size()

# %%
y = torch.tensor([[2], [1]])
y.size()

# %%
torch.mm(x, y).item()

# %%
x.dot(y)

# %%
# a = torch.diag(torch.tensor([2, 3], dtype = torch.double))
a = torch.tensor([[1, 2], [3, 4]], dtype = torch.double)
e, v = torch.eig(a, eigenvectors = True)

# %%
e

# %%
torch.eq(torch.tensor([[1.0, 2], [3, 4]], dtype = torch.double), torch.tensor([[1, 1], [3, 3]], dtype = torch.float))


# %%
x = torch.randn(4)
x

# %%
x.trunc()

# %%
aa = x.trunc()[3]
aa

# %%
aa * -1

# %%
x = torch.randn(4)
x

# %%
x.fill_(4)
x

# %%
x.erf_(0.1)

# %%
x

# %%
x = torch.rand(4)
x

# %%
x.erf()

# %%
x = torch.randn([1, 4])
x

# %%
x.flipud()

# %%
base = torch.arange(1, 5)
exp = torch.arange(2, 6)

base, exp, torch.pow(base, exp)

# %%
x = torch.tensor([1, 2, 3, 4, 5], dtype = torch.float)

# %%
torch.fmod(x, 1.5)

# %%
x.dtype

# %%
x = torch.tensor([-1.23, 1.5, 0.9, -3.14])

# %%
torch.equal(x.fix() + x.frac(), x)

# %%
a = torch.tensor([5, 10, 15])
b = torch.tensor([3, 4, 5])
torch.gcd(a, b)

# %%
c = torch.tensor([3])
torch.gcd(a, c)

# %%
a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[1, 1], [4, 4]])
torch.le(a, b) == torch.ge(a, b)

# %%
x = torch.randn(4)
x

# %%
x = torch.empty(10, 10).bernoulli(p = 0.5)

torch.sum(x)

# %%
torch.histc(torch.tensor([1., 2, 3]), bins = 4, min = 0, max = 0)

# %%
x = torch.arange(1, 10).reshape(3, 3)
index = torch.tensor([0, 2])
x.index_fill(dim = 0, value = -1, index = index)

# %%
x

# %%
index = [torch.tensor([0]), torch.tensor([1])]
x.index_put(indices = index, values = torch.tensor([10]))

# %%
x

# %%
x.size()

# %%
x = torch.dot(torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]))
x

# %%
x.size()

# %%
x = torch.randn(4)
x

#%%
torch.flip(x, dims = [0])

# %%
x = torch.rand([4, 4])
x_ = x.inverse()

# %%
x_

# %%
I = torch.mm(x, x_)
I

# %%
torch.allclose(I, torch.diag(torch.ones(I.shape[0])), rtol = 0, atol = 0.001)

# %%
x.inverse()

# %%
I = torch.mm(x, x.inverse())
torch.close(I, torch.diag(torch.ones(I.shape[0])), atol = 0.00001)

# %%
I.isclose(torch.diag(torch.ones(I.shape[0])), atol = 0.00001)

# %%
torch.isinf(torch.tensor([1, float("nan"), -1, float("inf"), float("-inf")]))

# %%
x = torch.tensor([1, 1+1j], dtype = torch.complex64)
x

# %%
x.dtype

# %%
x.isreal()


# %%
a = torch.tensor([5, 10, 15])
b = torch.tensor([3, 4, 5])
torch.lcm(a, b)

# %%
x = torch.tensor([1, 2 ,3], dtype = torch.double)
x.lgamma()

# %%
import numpy as np 
np.log(100)

# %%
np.log(100000000)

# %%
x = torch.randn(4)

# %%
a = torch.randn(4, 4)
a

# %%
a.matrix_power(3)

# %%
a = torch.tensor([[0.7, 0.2, 0.1], [0.6, 0.2, 0.2], [0.8, 0.15, 0.05]])
a

# %%
a.matrix_power(10)


# %%
x = torch.randn(4, 4)
x.median()

# %%
x

# %%
a = torch.tensor([[1, 2], [3., float("nan")]])
a

# %%
a.nansum()

# %%
a

# %%
a.dim()

# %%
a = torch.randn(10)
a.prod()

# %%
torch.remainder(torch.tensor([-3., -2, -1, 1, 2, 3]), -2)

# %%
a = torch.tensor([0, np.log(1 / 3)], dtype = torch.double)
a.sigmoid()

# %%
a = torch.empty(3, 4)
a

# %%
x = torch.randn(2, 1, 3)
x

# %%
x.squeeze_()

# %%
x.size()

# %%
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
x

# %%
x.tile((2, 1))

# %%
a = torch.tensor([1, 2, 1, 3, 4, 2, 1])
a

# %%
a.unique()

# %%
x = torch.zeros(5)
y = torch.tensor([1, 2, 2.71728, float("inf"), float("nan")])
torch.xlogy(x, y)

# %%

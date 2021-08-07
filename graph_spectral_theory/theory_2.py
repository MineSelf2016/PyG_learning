# %%
import numpy as np 

# %%
X = [[3, -1, -1, -1],
     [-1, 2, -1, 0],
     [-1, -1, 3, -1],
     [-1, 0, -1, 2]]
X = np.matrix(X)

# D = [

# ]

# D = np.matrix(D)

# A = [

# ]

# A = np.matrix(A)

# X = D - A



print("Matrix X: ")
print(X)
 
# %%
print()
print("Next, we caculate the eigenvalues and the corresponding eigenvectors")
eigenvalue, eigenvector = np.linalg.eig(X)

print()
print("eigenvalues: ")
print( eigenvalue)

print()
print("eigenvectors: ")
print(eigenvector)

# %%
a = eigenvector[:, 0]
b = eigenvector[:, 1]
c = eigenvector[:, 2]
d = eigenvector[:, 3]

#%%
print()
print("Next, we verify the orthdox of the eigenvectors: ")

for i in range(len(eigenvector)):
    print("the mode of vector ", i, " is: ", np.linalg.norm(eigenvector[:, i]))

print()

for i in range(len(eigenvector) - 1):
    for j in range(i + 1, len(eigenvector)):
        print("the dot of vector ", i, " and vector ", j, " is: ", np.dot(eigenvector[:, i].T, eigenvector[:, j]))




# %%
print(eigenvector.T.dot(eigenvector))


# %%
f = np.matrix([[1], [2], [3], [4]])

f
# %%
f_t = eigenvector.T.dot(f)
f_t

# %%
f_t_t = eigenvector.dot(f_t)
f_t_t

# %%

import numpy as np

import matplotlib.pyplot as plt
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import load_digits
digits = datasets.load_digits()

#Display the first digit
plt.figure(1, figsize=(4, 4))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')

plt.title("Target: 0")
print(digits.data.shape)
print(digits.data[0].reshape(8,8))

# plot 3D figure using Principal component analysis (PCA)
#Linear dimensionality reduction using Singular Value Decomposition of the data to project it to a lower dimensional space.
# 
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
proj = pca.fit_transform(digits.data)
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')

plt.title("components = 3")
p = ax.scatter(proj[:, 0], proj[:, 1],  proj[:, 2], cmap="Paired", c=digits.target)#
plt.colorbar(p)
plt.show()



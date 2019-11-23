import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

digits = datasets.load_digits()

#Display the first digit
plt.figure(1, figsize=(4, 4))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')

plt.title("Target: 0")
print(digits.data.shape)
print(digits.data[0].reshape(8,8))
plt.show()



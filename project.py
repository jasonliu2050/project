import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection  import cross_val_score, cross_val_predict
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
#from keras.datasets import mnist
digits = datasets.load_digits()
print(np.shape(digits.images))

n_samples = len(digits.images)

print("len=",n_samples)
data = digits.images.reshape((n_samples, -1))
print(data.shape)
# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)    #gamma=0.001  "scale"
knn = neighbors.KNeighborsClassifier(n_neighbors=2)

X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.45, random_state=0)
#(X_train,y_train), (X_test, y_test) = mnist.load_data()
# We learn the digits on the first half of the digits
classifier.fit(X_train, y_train)   #(data[:n_samples // 2], digits.target[:n_samples // 2])
knn.fit(X_train, y_train)

# Now predict the value of the digit on the second half:
expected = y_test #digits.target[n_samples // 2:]
predicted = classifier.predict(X_test) #(data[n_samples // 2:])
predicted1 = knn.predict(X_test)

print("Classification report for classifier %s:\n%s\n"
      % (classifier, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted))

print("Classification report for classifier %s:\n%s\n"
      % (knn, metrics.classification_report(expected, predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, predicted1))


plt.figure()

from sklearn.decomposition import PCA

# Apply PCA by fitting the data with only 8 dimensions
pca = PCA(n_components= 8).fit(digits.data)
# Transform the data using the PCA fit above
X = pca.fit_transform(digits.data)
y = digits.target

plt.scatter(X[:, 0],X[:, 1], c=y, cmap="Paired")
plt.colorbar()


plt.show()


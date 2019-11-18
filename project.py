import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection  import cross_val_score, cross_val_predict
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
#from keras.datasets import mnist
digits = datasets.load_digits()
print(np.shape(digits.images))



# The data that we are interested in is made of 8x8 images of digits, let's
# have a look at the first 4 images, stored in the `images` attribute of the
# dataset.  If we were working from image files, we could load them using
# matplotlib.pyplot.imread.  Note that each image must have the same size. For these
# images, we know which digit they represent: it is given in the 'target' of
# the dataset.
# images_and_labels = list(zip(digits.images, digits.target))
# for index, (image, label) in enumerate(images_and_labels[:4]):
#     plt.subplot(2, 4, index + 1)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Training: %i' % label)

# To apply a classifier on this data, we need to flatten the image, to
# turn the data in a (samples, feature) matrix:
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


# images_and_predictions = list(zip(digits.images[n_samples // 2:], predicted))
# for index, (image, prediction) in enumerate(images_and_predictions[:4]):
#     plt.subplot(2, 4, index + 5)
#     plt.axis('off')
#     plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
#     plt.title('Prediction: %i' % prediction)
#

plt.figure()

from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# proj = pca.fit_transform(digits.data)
# plt.scatter(proj[:, 0], proj[:, 1], c=digits.target, cmap="Paired")
# plt.colorbar()

# Apply PCA by fitting the data with only 8 dimensions
pca = PCA(n_components= 8).fit(digits.data)
# Transform the data using the PCA fit above
X = pca.fit_transform(digits.data)



y = digits.target

plt.scatter(X[:, 0],X[:, 1], c=y, cmap="Paired")
plt.colorbar()



plt.figure()
predicted1 = cross_val_predict(knn, X_test, y_test, cv=8)
plt.scatter(y_test, predicted1, label="Cross Validation")
plt.scatter(y_test, predicted, label="CNN")
# plt.scatter(y_test, predicted, label="Cross Validation")
# plt.scatter(y_test, predicted_values, label="Linear Regression")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
scores = cross_val_score(knn, X_train, y_train, cv=3)
print("Cross - validated scores:", scores)




plt.legend()
plt.show()

# plt.figure()
#
# from sklearn.decomposition import PCA
# pca = PCA() #(n_components=2)
# proj = pca.fit_transform(digits.data)
# plt.scatter(proj[:, 0], proj[:, 1], c=digits.target, cmap="Paired")
# plt.colorbar()
#
# plt.show()
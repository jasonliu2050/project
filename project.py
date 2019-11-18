import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection  import cross_val_score, cross_val_predict
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics

# Load dataset from online
digits = datasets.load_digits()

# Explore the dataset. Apply PCA by fitting the data with only 8 dimensions
plt.figure()
from sklearn.decomposition import PCA
pca = PCA(n_components= 8).fit(digits.data)
# Transform the data using the PCA fit above
X = pca.fit_transform(digits.data)
y = digits.target
plt.scatter(X[:, 0],X[:, 1], c=y, cmap="Paired")
plt.colorbar()

#To apply a classifier on this data, we can flat the image, to turn the data in a (samples, feature) matrix, or directly use the data.
n_samples = len(digits.images)
print("len=",n_samples)
data = digits.images.reshape((n_samples, -1))
print(data.shape)

#create 3 classifiers, print Classification Report
from sklearn.linear_model import Perceptron
classifiers = [
    ("Perceptron", Perceptron(tol=1e-3)),
    ("KNeighborsClassifier", neighbors.KNeighborsClassifier(n_neighbors=2)),
    ("SVC", svm.SVC(gamma=0.001))
]
for name, clf in classifiers:
    X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.35, random_state=0)
    clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)

    print("Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, predicted)))
    #print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, predicted))

# Showing how different algorithms perform on the hand-written digits dataset.
plt.figure()
heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
rounds = 20
digits = datasets.load_digits()
X, y = digits.data, digits.target

classifiers = [
    ("Perceptron", Perceptron(tol=1e-3)),
    ("KNeighborsClassifier", neighbors.KNeighborsClassifier(n_neighbors=2)),
    ("SVC", svm.SVC(gamma=0.001))
]
xx = 1. - np.array(heldout)
for name, clf in classifiers:
    print("training %s" % name)
    rng = np.random.RandomState(42)
    yy = []
    for i in heldout:
        yy_ = []
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=i, random_state=rng)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            yy_.append(1 - np.mean(y_pred == y_test))
        yy.append(np.mean(yy_))
    plt.plot(xx, yy, label=name)
plt.legend(loc="upper right")
plt.xlabel("Proportion train")
plt.ylabel("Test Error Rate")

plt.show()


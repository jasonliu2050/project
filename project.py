import numpy as np
import matplotlib.pyplot as plt
import keras

from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import  metrics
from keras.datasets import mnist
from sklearn.decomposition import PCA
from keras.layers import Dense # Dense layers are "fully connected" layers
from keras.models import Sequential # https://keras.io/models/sequential/
from sklearn.linear_model import Perceptron

classifiers = [
    ("Perceptron", Perceptron(tol=1e-3)),
    ("KNeighborsClassifier", neighbors.KNeighborsClassifier(n_neighbors=2))

]
image_vector_size = 28*28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)


# Use 10000 training samples
X, y = x_train[:10000], y_train[:10000]
# USe 100 components
pca = PCA(n_components = 100).fit(X)
projected = pca.fit_transform(X)


classifiers = [
    ("Perceptron", Perceptron(tol=1e-3)),
    ("KNeighborsClassifier", neighbors.KNeighborsClassifier(n_neighbors=2))

]
yy = []
yy_ = []
for name, clf in classifiers:
    print("Train %s" % name)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=21)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # yy_.append(1 - np.mean(y_pred == y_test))
    yy.append(np.mean(y_pred == y_test))

    print("Classification report for classifier %s:\n%s\n" % (clf, metrics.classification_report(y_test, y_pred)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_pred))


num_classes = 10
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
image_size = 784 # 28*28
num_classes = 10 # ten unique digits

model = Sequential()
model.add(Dense(units=1024, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=num_classes, activation='softmax'))

print(model.summary())

print("Please wait ... coming soon")
model.compile(optimizer="sgd", loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=10, verbose=False, validation_split=.1)
loss, accuracy  = model.evaluate(x_test, y_test, verbose=False)

# accuracy = format(accuracy, '.3f')
yy.append(accuracy)

x = ['','Perceptron', 'KNN', 'Neural Network']
xx =  [1, 2,3]

x_pos =  [i for i, _ in enumerate(x)]

plt.bar( xx, yy, color=["g","b","y"],width = 0.5)

plt.ylabel("Accuracy")
plt.title("Comparison of Machine learning Algorithm")

plt.xticks(x_pos, x)

plt.show()

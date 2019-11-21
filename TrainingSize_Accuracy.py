import keras
from keras.datasets import mnist
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn import datasets, metrics
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt

# Setup train and test splits
(x_train, y_train), (x_test, y_test) = mnist.load_data()
image_vector_size = 28*28

x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)

Sample_size = [30,100,200,500,1000,2000,5000,10000]
Accuracy =[]
Accuracy1=[]
for samples in Sample_size:
    clf = neighbors.KNeighborsClassifier(n_neighbors=2)
    clf1 = Perceptron(tol=1e-3)
    clf.fit(x_train[:samples], y_train[:samples])
    predicted = clf.predict(x_test[:2000])

    clf1.fit(x_train[:samples], y_train[:samples])
    predicted1 = clf1.predict(x_test[:2000])

    accuracy =  metrics.classification_report(y_test[:2000], predicted[:2000], output_dict=True)
    accuracy1 = metrics.classification_report(y_test[:2000], predicted1[:2000], output_dict=True)
    Accuracy.append(accuracy["accuracy"])
    Accuracy1.append(accuracy1["accuracy"])
    print("Please wait ...")
plt.title("Classifier Accuracy & Number of Training Samples")
plt.plot(Sample_size, Accuracy)
plt.plot(Sample_size, Accuracy1)
plt.xlabel("Number of Training Samples")
plt.ylabel("Accuracy")
plt.legend(['KNN', 'Perceptron'], loc='best')

plt.show()

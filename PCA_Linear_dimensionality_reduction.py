import numpy as np
# import keras
# import seaborn as sns
# import pandas as pd
# import matplotlib.pyplot as plt

from keras.datasets import mnist
from sklearn import neighbors

from sklearn import datasets, metrics
from sklearn.decomposition import PCA

image_vector_size = 28*28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], image_vector_size)
x_test = x_test.reshape(x_test.shape[0], image_vector_size)

print("please wait ...")
# pca = PCA(n_components = 500).fit(x_train)  # Accuracy will be less than 0.94,
# pca = PCA(n_components = 200).fit(x_train) # Accuracy will be 0.94

pca = PCA(n_components = 60).fit(x_train) # Accuracy will be 0.96

x_train = pca.transform(x_train)
x_test = pca.transform(x_test)

clf = neighbors.KNeighborsClassifier(n_neighbors=2)
clf.fit(x_train, y_train)
predicted = clf.predict(x_test[:2000])
report =metrics.classification_report(y_test[:2000], predicted[:2000])
report_dict =  metrics.classification_report(y_test[:2000], predicted[:2000], output_dict = True)
print("Classification report for classifier %s:\n%s\n" % (clf, report ))



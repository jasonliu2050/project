# cebd1160: Digital Image Handwriting Recognition
Instructions for final projects.

| Name | Date |
|:-------|:---------------|
|Jun Liu | 2019.11.18|

-----

## Resources
This Project repository includes the following items:

- Python scripts for analysis:  
 [ExploreDigitsDataset](https://github.com/jasonliu2050/project/blob/master/ExploreDigitsDataset.py)  
 [Keras_Neural_Network](https://github.com/jasonliu2050/project/blob/master/Keras_Neural_Network.py)  
 [Training sample size & Model Accuracy](https://github.com/jasonliu2050/project/blob/master/TrainingSize_Accuracy.py)  
 [Scikit-learn algorithm comparison](https://github.com/jasonliu2050/project/blob/master/project.py)  
- Results/Pictures:  
 [Scores](https://github.com/jasonliu2050/project/blob/master/figures/Scores.png)
- Dockerfile for your experiment  
  TBD
- runtime-instructions:   
  RUNME.md
-----

## Research Question

How to solve handwriting digits recognition problem using machine learning algorithms?
 
### Abstract
The problem we need to solve is to classify handwritten digits. The goal is to take an image of a handwritten digit and determine what that digit is. The digits range from 0 through 9. We could apply machine learning algorithms to solve our problem. Using simple handwriting digit datasets provided by Scikit-learn, we achieved from 97% F1-score with Nearest Neighbor (KNN) classifier, to achieving 99% F1 score with Support Vector Classifier(SVC). The scope of this article also include comparing the different classifiers, using dataset from the famous MNIST (Modified National Institute of Standards and Technology), and try to achieve higher performance by choosing parameters along with dataset preprocessing technique.
### Introduction
When use public digit dataset provided from scikit-learn, the data that we are interested in is made of 8x8 image, it contains 1797 samples, 64 features. Each sample in the dataset represent an image that is 8 pixels in height and 8 pixels in width, the total of 64 pixels. Each image is labelled with their corresponding category that is the actual digit from 0 to 9 for a total of 10 different type of labels. Using these data, we could find out the relationships between image pixels and their digit values, which can then be used for predicting the target digit. 

Scikit-learn machine learning algorithms: Support Vector Machines (SVMs), Nearest Neighbor (NN) techniques and Perceptron are used in our solution. For Scikit-learn simple digit dataset, all these algorithms performs very well without doing feature selection or any preprocess for that dataset, for this reason, we also use MNIST training digit dataset, which is much more larger compair to Scikit-learn training Digits set, it contains 60,000 samples in total and 784 features. Each sample in the dataset represent an image that is 28 pixels in height and 28 pixels in width, hence the total of 784 pixels. Keras Sequential Neural Networks algorithm is also used for this training set. 

### Methods Using Scikit-learn
For Scikit-learn training set, we use Support Vector Machines (SVMs) Nearest Neighbor (NN) and Perceptron techniques to solve the problem. The tasks involved are the following:

1. Load and Explore the Digit Dataset
2. Simple visualization and classification of the digits dataset
3. Dataset Preprocessing 
4. Train a classifier, and test its accuracy

#### 1. Load and Explore the Dataset

![matrix](./figures/ExploreDataset.png)


#### 2. Simple visualization and classification of the digits dataset

![matrix](./figures/PrincipalComponentAnalysis.png)

#### 3. Dataset preprocessing
I give a example for data preprocessing when use Nearest Neighbor (KNN) classifier.

The accuracy of KNN can be severely degraded with high-dimension data because there is little difference between the nearest and farthest neighbor. Dimensionality reduction techniques like PCA could be executed prior to appplying KNN and help make the distance metric more meaningful.

Since the original dimension is quite large (784 input features), the dimensionality reduction becomes necessary. First, we extract the principal components from the original data. We do this by fitting a Principle Component Analysis (PCA) on the training set, then transforming the data using the PCA fit. We used the PCA module of the scikit-learn Python library with n_components set to differenct value to transform the dataset. From the test result, I found the first 30 ~ 60 principal components can interpret approximately 90% of total information, which suffice to be representative of the information in the original dataset. We thus choose the first 60 principal components as the extracted features. The test result shows Accuracy are much better than use all input features. ([Test Script: PCA Linear Dimensionality Reduction ](https://github.com/jasonliu2050/project/blob/master/PCA_Linear_dimensionality_reduction.py))  
The following picture show Training data size are very important to the final test accuracy result:[Reference Training sample size & Model Accuracy](https://github.com/jasonliu2050/project/blob/master/TrainingSize_Accuracy.py)    
![matrix](./figures/Training_Size_Accuracy.png)

### 4. Final Test Result

#### Methods Using Scikit-learn Algorithms
Picture below showing difference algorithms perform on the Scikit-learn hand-written digits dataset.
The test result are using Scikit-learn toy dataset. All test result are very good.

![matrix](./figures/Comparation.png)

#### Methods Using Keras Sequential Neural Networks

Keras Neural Networks are very diferent from Scikit-learn, we will not give more details in this article, you can check the link below for reference. Here we can see from script test result, without doing any optimazition, Keras Neural Network algorithm easily reach 0.95 accuracy on MNIST training digit dataset, which is better than Scikit-learn when use the same test dataset.

![matrix](./figures/KerasScores.png)

## Discussion
The methods used above did solved the problem of identifying handwritten digits. These methods shows that using the current online training dataset, all  Scikit-learn algorithms: SVC, KNN, Perceptron performs very good when using the small dataset. when using MNIST training digit dataset, Keras Neural Network algorithm has the best performance.    
We still need to test these algorithm performance use digit image in real life, for example, the size may changes, the digit may rotate in different direction, and how to handle digital image samples in dark backgroud such as a car plate, those new problems need use Computer Vision and Pattern Recognition algorithm, such as OpenCV. All these issues are not tested because of the our dataset limit.  

## References
[Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits)  
[Scikit-learn](https://scikit-learn.org/stable/whats_new.html#version-0-21-3)  
[Keras Example](https://medium.com/tebs-lab/how-to-classify-mnist-digits-with-different-neural-network-architectures-39c75a0f03e3)  
[Keras Sequential model](https://keras.io/models/sequential/)  
[Matplotlib3.1.1](https://matplotlib.org/3.1.1/users/whats_new.html)  
[Handwriting Article reference](https://medium.com/the-andela-way/applying-machine-learning-to-recognize-handwritten-characters-babcd4b8d705)  
[Comparing various online solvers](https://scikit-learn.org/stable/auto_examples/linear_model/plot_sgd_comparison.html#sphx-glr-auto-examples-linear-model-plot-sgd-comparison-py)  



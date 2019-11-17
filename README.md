# cebd1160: Digital Image Handwriting recognition
Instructions for final projects.

| Name | Date |
|:-------|:---------------|
|Jun Liu | 2019.11.18|

-----

### Resources
Your repository should include the following:

- Python script for your analysis: project.py
- Results figure/saved file
- Dockerfile for your experiment
- runtime-instructions in a file named RUNME.md

-----

## Research Question

How can we recognise digit in handwriting digital image (8x8 pixels)? 
### Abstract
The problem we need to solve is to classify handwritten digits. The goal is to take an image of a handwritten digit and determine what that digit is. The digits range from 0 through 9. By using handwriting digit dataset provided from Sklearn, we could apply machine learning algorithm to solve the problem, with different classifiers and parameters along with preprocessing techniques from XXX% error rate with linear classifier to achieving XXX% error rate with Nearest Neighbor (NN). The scope of this article include to compare the different classifiers with different parameters and try to achieve higher performance.

### Introduction
We will use public digit dataset provided from Sklearn, the data that we are interested in is made of 8x8 images of digits, it contains 1797 samples, 64 features. Each sample in the dataset represent an image that is 8 pixels in height and 8 pixels in width, the total of 64 pixels. Each image is labelled with their corresponding category that is the actual digit from 0 to 9 for a total of 10 different labels. Using these data, we could find out the relationships between image pixels and their digit values, which can then be used for predicting the target digit. Sklearn machine learning algorithm Support Vector Machines (SVMs) and Nearest Neighbor (NN) techniques can be used to solve the problem.
To apply a classifier on this data, we need to flatten the image, to
turn the data in a (samples, feature) matrix:

### Methods
We use Support Vector Machines (SVMs) and Nearest Neighbor (NN) techniques to solve the problem. The tasks involved are the following:
1. Load Digit Dataset (sklearn recommended ways to load datasets)
2. Train a classifier that can categorize the handwritten digits
3. Apply the model on the test set and report its accuracy
Based on the preliminary performance of this regressor, we found that the current model didn't provide consistent performance, but shows promise for success with more sophisticated methods.

- pseudocode for this method (either created by you or cited from somewhere else)
- why you chose this method

### Results

Brief (2 paragraph) description about your results. Include:

- At least 1 figure
- At least 1 "value" that summarizes either your data or the "performance" of your method
- A short explanation of both of the above

### Discussion
Brief (no more than 1-2 paragraph) description about what you did. Include:

- interpretation of whether your method "solved" the problem
- suggested next step that could make it better.

### References
Dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html#sklearn.datasets.load_digits
Sklearn: https://scikit-learn.org/stable/whats_new.html#version-0-21-3
Seaborn: https://seaborn.pydata.org/index.html
Matplotlib3.1.1: matplotlib.org/3.1.1/index.html
Handwriting Article reference:
https://medium.com/the-andela-way/applying-machine-learning-to-recognize-handwritten-characters-babcd4b8d705
https://en.wikipedia.org/wiki/Handwriting_recognition

-------

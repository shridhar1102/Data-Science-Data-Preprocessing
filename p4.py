#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:01:34 2018


Description of the project: Establishing the realationship between x and y using algorithms such as K-Nearest-Neighbors and
and Support Vector Machines.
dataset consiting of training observations (x,y) and would like to capture the relationship between x
. More formally, our goal is to learn a function h:X→Y.so that given an unseen observation x can confidently predict the corresponding output y

"""
# import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
y = iris.target


#Splitting the dataset into the training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting the classifier to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 8, metric = 'minkowski', p = 2)

#Create your classifier here
classifier.fit(X_train, y_train)

# Prediciting the resr set results
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm_knn = confusion_matrix(y_test, y_pred)

# Visualizing the training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train

x1 = np.arange(X_set[:, 0].min(), X_set[:, 0].max(), step = 0.01)
x2 = np.arange(X_set[:, 1].min(), X_set[:, 1].max(), step = 0.01)
X1, X2= np.meshgrid(x1,x2)


Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)

plt.contourf(X1, X2, Z.reshape(X1.shape), alpha = 0.55,
             cmap = ListedColormap(('red', 'green','blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for j in np.unique(y_set):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(j), label = j)

plt.title('KNN (Training set)')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

# Visualizing the test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test

x1 = np.arange(X_set[:, 0].min(), X_set[:, 0].max(), step = 0.01)
x2 = np.arange(X_set[:, 1].min(), X_set[:, 1].max(), step = 0.01)

X1, X2 = np.meshgrid(x1,x2)


Z = classifier.predict(np.array([X1.ravel(), X2.ravel()]).T)

plt.contourf(X1, X2, Z.reshape(X1.shape), alpha = 0.55,
             cmap = ListedColormap(('red', 'green', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for j in np.unique(y_set):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(j), label = j)

plt.title('KNN (Test set)')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()




#Support Vector Machine
# Fitting the classifier to the Training set
from sklearn.svm import SVC

classifier_svm=SVC(kernel='linear')


#Create your classifier here
classifier_svm.fit(X_train, y_train)

# Prediciting the resr set results
y_pred = classifier_svm.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(y_test, y_pred)

# Visualizing the training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train

x1 = np.arange(X_set[:, 0].min(), X_set[:, 0].max(), step = 0.01)
x2 = np.arange(X_set[:, 1].min(), X_set[:, 1].max(), step = 0.01)
X1, X2= np.meshgrid(x1,x2)


Z = classifier_svm.predict(np.array([X1.ravel(), X2.ravel()]).T)

plt.contourf(X1, X2, Z.reshape(X1.shape), alpha = 0.55,
             cmap = ListedColormap(('red', 'green','blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for j in np.unique(y_set):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green', 'blue'))(j), label = j)

plt.title('SVM (Training set)')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

# Visualizing the test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test

x1 = np.arange(X_set[:, 0].min(), X_set[:, 0].max(), step = 0.01)
x2 = np.arange(X_set[:, 1].min(), X_set[:, 1].max(), step = 0.01)

X1, X2 = np.meshgrid(x1,x2)


Z = classifier_svm.predict(np.array([X1.ravel(), X2.ravel()]).T)

plt.contourf(X1, X2, Z.reshape(X1.shape), alpha = 0.55,
             cmap = ListedColormap(('red', 'green', 'blue')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for j in np.unique(y_set):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green','blue'))(j), label = j)

plt.title('SVM (Test set)')
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()
plt.show()

"""
How many observations are in this data set ? (2 points)
Ans-There are 150 obersvation in the dataset

How many features are in this data set ? (2 points)
Ans-4 features available  in the dataset and we have taken 2 features for prediction analysis.


Compare the confusion matrix for both KNN and linear SVM, which algorithm get the better result (6 points)
Ans- For KNN correct classifications are 27 and total classifications are 38 so accuaracy is 27/38=0.71 or TP/(TP+FP)=27/(27+11)=0.71

For SVM correct classification are 29 and total classifications are 38 so accuracy is 29/38=0.76 and or TP/(TP+FP) 29/(29+5)= 0.76

So SVM has better accuracy than KNN classification





"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 17:09:59 2018


Description of the Project- Preparing the prediction model using Linear regression class  for the dataset imported.
Steps are implemented below to prepare the model.
Displayed the visualisation of scatter plot.
"""
#Importing required packages
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


#Read the dataset
dataset=pd.read_excel('autoInsurance.xls')
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,1].values

#splitting the data into Trained and Test set
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3, random_state=0)

#regressor object
regressor=LinearRegression()
regressor.fit(X_train, Y_train)

#Predicting the Test set results
y_pred=regressor.predict(X_test)

# visualsing the data of trained set
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Payment vs  claims (Training set)')
plt.xlabel('number of claims')
plt.ylabel('total payment for all the claims.')
plt.show()

# visualsing the data of test set

plt.scatter(X_test, Y_test, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Payment vs claims(Test set)')
plt.xlabel('number of claims')
plt.ylabel('total payment for all the claims.')
plt.show()



"""
How many observations are in this data set
Ans-63 observations

How many features are in this data set ?
Ans-1

What is the response for this data set ?
Ans-total payment for all the claims(Y) is response for this data set.










"""

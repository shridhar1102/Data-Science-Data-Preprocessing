#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 07:23:26 2018




#Taking care of missing values (Replace missing values with mean)
dataset= dataset.replace('?', np.NaN)
imputer = Imputer(missing_values= 'NaN', strategy='mean', axis=0)
imputer.fit(dataset)#fit the data to dataset
dataset_clean = imputer.transform(dataset)

#spliting the data into Training set and Test set-dataset is split to training part (80%) and testing part (20%)
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:, -1].values
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2, random_state=0)


#Description of the data matrix
'''
• How many observations are in this data set ?
Ans-There are 452 observations or instances in this dataset.

• How many features are in this data set ?
Ans-Theres are 279 features or attributes in this dataset

What is the response for this data set ?
Ans-Last column(Y) of the dataset is response for this dataset.

'''

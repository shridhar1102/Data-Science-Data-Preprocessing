#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 12:11:24 2018

@author: shridhar
"""

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm


dataset=pd.read_csv('50_Startups.csv')
dataset=pd.get_dummies(dataset,columns=['State'])
X=dataset.iloc[:,[0,1,2,4,5]].values
y=dataset.iloc[:,3].values

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25, random_state=0)

regressor=LinearRegression()
regressor.fit(X_train, y_train)
#Predicting the Test set results
y_pred=regressor.predict(X_test)

# Add one column of ones to your data matrix X, make it as the first column of X
X=np.append(arr=np.ones((50,1)), values=X, axis =1)

#Select all predictors from X, and apply it with OLS.
X_opt= X[:,[0,1,2,3,4,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt= X[:,[0,1,2,3,5]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt= X[:,[0,1,2,3]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt= X[:,[0,1,3]] #adj. R-sqared=0.948
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

X_opt= X[:,[0,1]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

def backwardElimination(x, sl):
    numVars=len(x[0])
    for i in range(0, numVars):
        regressor_OLS=sm.OLS(y,x).fit()
        maxVar= max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0,numVars - i):
                if(regressor_OLS.pvalues[j].astype(float)== maxVar):
                    x = np.delete(x,j,1)
                    regressor_OLS.summary()
                    return x
                
                
SL=0.05
X_opt = X[:, [0,1,2,3,4,5]]
X_Modeled=backwardElimination(X_opt, SL)

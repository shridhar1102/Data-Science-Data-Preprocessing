#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 00:07:33 2018

@author: shridhar
Description :calculate the posterior probability P(y=1|X=x) and P(y=2|X=x) in python by using the formula we discussed during the class.
Student name-Shridhar Kevati
Student ID-999992831
"""
#Import Libraries
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det

mu1 = np.matrix('5.01;3.42')
mu2 = np.matrix('6.26; 2.87')
cov1 = np.matrix('0.122 0.098; 0.098 0.142')
cov2 = np.matrix('0.435 0.121; 0.121 0.110')
X = np.matrix('6.75;4.25')

#prior probability
p1 = 0.33
p2 = 0.67
d=2 # No of Categories

# Calcualting f1(x) using formula
f1x=(1/(((np.sqrt(2*np.pi))**d)*(np.sqrt(det(cov1)))))*(np.exp((-(((np.transpose(X-mu1))*(inv(cov1))*(X-mu1))/2))))

# Calcualting f2(x) using formula
f2x=(1/(((np.sqrt(2*np.pi))**d)*(np.sqrt(det(cov2)))))*(np.exp((-(((np.transpose(X-mu2))*(inv(cov2))*(X-mu2))/2))))

#Calculate posterior probability
py1_1 = (f1x*p1)
py1_2 = (f1x*p1+f2x*p2)
py1 = py1_1/py1_2
print(' posterior probability for c1: ',py1)


#Calculate posterior probability
py2_1 = (f2x*p2)
py2_2 = (f1x*p1+f2x*p2)
py2 = py2_1/py2_2
print(' posterior probability for c2: ',py2)
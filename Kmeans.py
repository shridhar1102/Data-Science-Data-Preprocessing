#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 11:52:20 2018


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#read dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The Elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
kmeans = KMeans(n_clusters = 5, init='k-means++', random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#visualising the clusters
plt.scatter(X[y_kmeans ==0,0], X[y_kmeans ==0,1], s=50, c='red', label = 'Cluster 1')
plt.scatter(X[y_kmeans ==1,0], X[y_kmeans ==1,1], s=50, c='blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans ==2,0], X[y_kmeans ==2,1], s=50, c='green', label = 'Cluster 3')
plt.scatter(X[y_kmeans ==3,0], X[y_kmeans ==3,1], s=50, c='cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans ==4,0], X[y_kmeans ==4,1], s=50, c='black', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, c='yellow' , label = 'Centroids')
plt.title('Clusters of Customers')
plt.xlabel('Annual income(k$)')
plt.ylabel('Spending score(1-100)')
plt.legend()
plt.show()

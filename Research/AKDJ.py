#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 18:16:20 2018

@author: huzmorgoth
"""

from sklearn.cluster import KMeans
import pandas as pd
import pylab as pl
import numpy as np
import scipy.stats as stats 
import matplotlib.pyplot as plt
import sklearn
import re
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import neighbors
from mpl_toolkits.mplot3d import Axes3D
import sklearn.metrics as sm
from sklearn.metrics import confusion_matrix,classification_report

t_X = pd.read_csv("/Users/huzmorgoth/RODEO/shah/electricitydataset.csv")

X = t_X[['kwh','unit']]

y = t_X[['athome']]

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, y, test_size 
                                            = 0.33, random_state = 4)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

"""--------------------------------------------------------------"""

"""K-NN Regression"""

n_neighbors = 5

for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    
    # Fitting regression model and Predicting the results
    Y_ = knn.fit(X_train, Y_train).predict(X_test)
    
    # Plotting a graph
    plt.subplot(2, 1, i + 1)
    plt.scatter(X_train.mean(axis=1), Y_train, c='k', label='electricitydata')
    plt.plot(X_test.mean(axis=1), Y_, c='g', label='predictedElectData')
    plt.subplots_adjust(hspace=.5)
    plt.axis(ymin=0, ymax=3)
    plt.legend()
    plt.title("\nKNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,weights))
    
    # Accuracy of the results

    MSE = np.mean((Y_test - Y_) ** 2)
    print(MSE)

plt.show()


"""--------------------------------------------------------------"""

"""Linear Regression"""

LR = LinearRegression()


LR.fit(X_train, Y_train)

#pre_train = LR.predict(X_train)

# TESTING

pre_test = LR.predict(X_test)

print(Y_test.head(10))


#Mean Sqr Error

#MSE1 =  np.mean((Y_train - pre_train) ** 2)
MSE2 = np.mean((Y_test - pre_test) ** 2)

#print("Fit X_train, and calculate the error with Y_train:", MSE1)
print("Fit X_train, and calculate the error with X_test, Y_test:", MSE2) 

# Accuracy of the result

accu = np.average(Y_test) - np.average(pre_test)

print("Error of the result:(%)", accu)

# Plotting the residual graph

"""
plt.scatter(pre_train, pre_train - Y_train, c='b', s=10, alpha=1)
plt.scatter(t_X['unit'], t_X['kwh'], c='g', s=10)
plt.hlines(y=0, xmin=0.5, xmax=2.5)
plt.title("Residual Plot training(Blue) and test(Green)")
plt.ylabel("residuals")
"""

plt.scatter(X_test.mean(axis=1), Y_test, c='k', label='data')
plt.scatter(X_test.mean(axis=1), pre_test, c='r', label='prediction')
plt.axis(ymin=0, ymax=3)
plt.legend()
plt.title("Actual data vs Predicted results")
plt.show

"""-------------------------------------------------------------"""

"""KNN Clustering"""

plt.scatter(t_X['unit'], t_X['kwh'], c='g', s=10)
plt.title("Actual Data")
plt.ylabel("power")
plt.xlabel("units")

  """-------------------------------------"""
  
clustering = KMeans(n_clusters=3, random_state = 9)
clustering.fit(X)

col = np.array(['darkgray','lightsalmon','powderblue'])

plt.scatter(y=t_X[['kwh']], x=t_X[['unit']], c=col[clustering.labels_], s=10)
plt.axis('auto')
plt.title("K-means Classification")

relabel = np.choose(clustering.labels_,[2,0,1,3,4,5]).astype(np.int64)

plt.scatter(x=t_X[['unit']], y=t_X[['kwh']], c=col[relabel], s=10)
plt.axis('auto')
plt.title("K-means Classification")

print(classification_report(y, relabel))
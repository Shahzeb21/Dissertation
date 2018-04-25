#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 18:16:20 2018

@author: shahz
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
from sklearn.metrics import precision_score

t_X = pd.read_csv(r'''C:\Users\shahz\Documents\Research\electricitydatasetUnit.csv''')

X = t_X[['kwh','unit']]

y = t_X[['athome']]

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.33, random_state = 4)

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
    plt.plot(X_test.mean(axis=1), Y_, c='lightsalmon', label='predictedElectData')
    plt.subplots_adjust(hspace=.5)
    plt.axis(ymin=-0.5, ymax=2.5)
    plt.legend()
    plt.title("\nKNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,weights))
    
    # Accuracy of the results

    MSE = np.mean((Y_test - Y_) ** 2)
    print("MEAN SQUARE ERROE: " , MSE)

precision_score(Y_test, Y_.round(), average=None)
plt.show()


"""--------------------------------------------------------------"""

"""Linear Regression"""

LR = LinearRegression()


LR.fit(X_train, Y_train)

#pre_train = LR.predict(X_train)

# TESTING

pre_test = LR.predict(X_test)



print(Y_test.head(10))
print(pd.DataFrame(pre_test).head(10))
precision_score(Y_test, pre_test.round(), average=None)

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

plt.scatter(t_X['kwh'], t_X['unit'], c='lightsalmon', s=10)
plt.title("Actual Data")
plt.ylabel("Time /mins")
plt.xlabel("Power /kwh")

"""-------------------------------------"""
  
clustering = KMeans(n_clusters=3, random_state = 0)
clustering.fit(X)

col = np.array(['darkgray','lightsalmon','powderblue'])

plt.scatter(y=t_X[['kwh']], x=t_X[['unit']], c=col[clustering.labels_], s=10)
plt.axis('auto')
plt.title("K-means Classification")

relabel = np.choose(clustering.labels_,[2,0,1,3,4,5]).astype(np.int64)

plt.scatter(x=t_X[['unit']], y=t_X[['kwh']], c=col[relabel], s=10)
plt.axis('auto')
plt.title("K-means Classification")

# Converting numpyndarray array to pandas.dataframe to make it callable
callable = pd.DataFrame(Y_)

# Pinting first 20 values of the testing and predicted data
print("This is the testing data: ",Y_test.head(20))   
print("This is the predicted data: ",callable.head(10))

# calculating precision score of the algorithm
precision_score(Y_test, Y_.round(), average=None)

print(classification_report(y, relabel))


""" Statistics """
# https://en.wikipedia.org/wiki/Precision_and_recall

#program to calculate True positives and False positives and True negatives and False negatives

#Trying to figure out how to access the data sets
print(Y_test.iloc[2]['athome'])
print(callable.iloc[3][0])


# The total conditions that are positives
CP = 0
# The total conditions that are negatives
CN = 0
# When the person is infact home and predicted to be home
TP = 0
# When the person is not home but predicted to be home
FP = 0 
# When the person is not home and predicted to not be home
TN = 0
# When the person is home but predicted to not be home
FN = 0

for i in range (Y_test.size):
    # Calculating positive conditions
    if Y_test.iloc[i]['athome'] > 0:
        CP = CP + 1
    # Calculating negative conditions
    elif Y_test.iloc[i]['athome'] == 0:
        CN = CN + 1
        
        
    # Calculating true positives
    if Y_test.iloc[i]['athome'] == (callable.iloc[i][0]).round():
        TP = TP + 1
    # Calculating false positives
    elif (Y_test.iloc[i]['athome'] == 0) & ((callable.iloc[i][0]).round() > Y_test.iloc[i]['athome']):
        FP = FP + 1
    # calculating true negatives
    elif (Y_test.iloc[i]['athome']) == ((callable.iloc[i][0]).round()) == 0:
        TN = TN + 1
    # Calculating false negatives
    elif ((Y_test.iloc[i]['athome']) > 0) & ((callable.iloc[i][0]).round() == 0):
        FN = FN + 1
        
    
print(TP)
print(FP)
print(TN)
print(FN)
print(CP)
print(CN)
print(Y_test.size)


        



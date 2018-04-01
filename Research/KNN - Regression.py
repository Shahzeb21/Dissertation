# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 01:24:41 2018

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

"""K-NN Regression"""

n_neighbors = 5

for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    
    # Fitting regression model and Predicting the results
    Predicted = knn.fit(X_train, Y_train).predict(X_test)
    
    # Plotting a graph
    plt.subplot(2, 1, i + 1)
    plt.scatter(X_train.mean(axis=1), Y_train, c='k', label='electricitydata')
    plt.plot(X_test.mean(axis=1), Predicted, c='g', label='predictedElectData')
    plt.subplots_adjust(hspace=.5)
    plt.axis(ymin=0, ymax=3)
    plt.legend()
    plt.title("\nKNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors,weights))
    
    # Accuracy of the results

    MSE = np.mean((Y_test - Predicted) ** 2)
    print("MEAN SQUARE ERROE: " , MSE)

precision_score(Y_test, Predicted, average=None)
plt.show()

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

callable = pd.DataFrame(Predicted)

precision_score(Y_test, Y_.round(), average=None)

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
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 01:07:16 2018

@author: shahz
"""

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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn import neighbors
from mpl_toolkits.mplot3d import Axes3D
import sklearn.metrics as sm
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import precision_score

#t_X = pd.read_csv(r'''C:\Users\shahz\Documents\Research\electricitydatasetUnit.csv''')
t_X = pd.read_csv(r'''C:\Users\shahz\Documents\303 - RESEARCH 2018\Dissertation\Research\electricitydatasetUnit.csv''')

X = t_X[['kwh','unit']]

y = t_X[['athome']]

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.33, random_state = 4)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

"""--------------------------------------------------------------"""

def AccPerc(Training , Prectiction):

    PredictedY = pd.DataFrame(Prectiction)
    
    count = 0
    for i in range (Training.size):
        #print(Y_test.iloc[i]['athome'], ' - ' , PredictedY.iloc[i][0])
        if (Training.iloc[i]['athome'] == PredictedY.iloc[i][0]):
            count = count + 1
            
    percentage = 100*(count/Training.size)
    pres_scr = precision_score(Training, Prectiction.round(), average='weighted')
    print('The precision score: ' , pres_scr)
    print('The algorithm predicted ', count ,'/', Training.size , ' correctly')
    print('That is an accuracy percentage off: ', percentage )
    
def AccPercLinear(Training , Prectiction):

    PredictedY = pd.DataFrame(Prectiction)
    
    count = 0
    for i in range (Training.size):
        #print(Y_test.iloc[i]['athome'], ' - ' , PredictedY.iloc[i][0])
        if (Training.iloc[i]['athome'] == (PredictedY.iloc[i][0]).round()):
            count = count + 1
            
    percentage = 100*(count/Training.size)
    pres_scr = precision_score(Training, Prectiction.round(), average='weighted')
    print('The precision score: ' , pres_scr)
    print('The algorithm predicted ', count ,'/', Training.size , ' correctly')
    print('That is an accuracy percentage off: ', percentage )

"""K-NN Regression"""

n_neighbors = 3

for i, weights in enumerate(['uniform', 'distance']):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    
    # Fitting regression model and Predicting the results
    Y_ = knn.fit(X_train, Y_train).predict(X_test)
    print(Y_)
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
print("MEAN SQUARE ERROR: " , MSE)

precision_score(Y_test, Y_.round(), average=None)
plt.show()

#Accuracy defining function called


AccPerc (Y_test,Y_)

k =0
K =[1]
AccuracyDist = []
AccuracyUnifrm = []
MSE_Dist = []
MSE_Unifrm = []
percentage_Dist = []
percentage_Unifrm = []

while k<156:
    k = k+3
    K.append(k)
    print(K)
    
for k in K:
    for i, weights in enumerate(['distance']):
        knn = neighbors.KNeighborsRegressor(k, weights=weights)
    
        # Fitting regression model and Predicting the results
        Y_ = knn.fit(X_train, Y_train).predict(X_test)
        
        PredictedY = pd.DataFrame(Y_)
        pres_scr = precision_score(Y_test, Y_.round(), average='weighted')
        AccuracyDist.append(pres_scr)
        MSE_Dist.append(np.mean((Y_test - Y_) ** 2))
        
        count = 0
        for i in range (Y_test.size):
            #print(Y_test.iloc[i]['athome'], ' - ' , PredictedY.iloc[i][0])
            if (Y_test.iloc[i]['athome'] == PredictedY.iloc[i][0]):
                count = count + 1
        percentage_Dist.append(100*(count/Y_test.size))

     
    for i, weights in enumerate(['uniform']):
        knn = neighbors.KNeighborsRegressor(k, weights=weights)
    
        # Fitting regression model and Predicting the results
        Y_ = knn.fit(X_train, Y_train).predict(X_test)
        PredictedY = pd.DataFrame(Y_)
        pres_scr = precision_score(Y_test, Y_.round(), average='weighted')
        AccuracyUnifrm.append(pres_scr)
        MSE_Unifrm.append(np.mean((Y_test - Y_) ** 2))
        
        count = 0
        for i in range (Y_test.size):
            #print(Y_test.iloc[i]['athome'], ' - ' , PredictedY.iloc[i][0])
            if (Y_test.iloc[i]['athome'] == PredictedY.iloc[i][0]):
                count = count + 1
        percentage_Unifrm.append(100*(count/Y_test.size))


print(AccuracyDist)
print(AccuracyUnifrm)
print(percentage_Dist)
print(percentage_Unifrm)

MSE_U = []
MSE_D = []
for i in range (0, len(MSE_Unifrm)):
    MSE_U.append(MSE_Unifrm[i]['athome'])
    MSE_D.append(MSE_Dist[i]['athome'])
    
print(MSE_U)
print(MSE_D)

plt.plot(K, AccuracyDist, 'purple', label='Distance')
plt.plot(K, AccuracyUnifrm, 'darkgreen', label='Uniform')

#Plot for the Precision Score against the number of nearest neigbours
#plt.plot(K,AccuracyDist,'ro')
plt.ylabel('Precision Score')
plt.xlabel('Values of "k"')
plt.legend()
plt.title("Precision Score against Nearest Neighbours")
plt.show()
print(len(K))

plt.plot(K, percentage_Dist, 'blue', label='Distance')
plt.plot(K, percentage_Unifrm, 'lightsalmon', label='Uniform')

#Plot for the Precision Score against the number of nearest neigbours
plt.ylabel('Accuracy')
plt.xlabel('Values of "k"')
plt.legend()
plt.title("Accuracy against Nearest Neighbours")
plt.show()
print(len(K))


#Plot for the MSE against the number of nearest neigbours
plt.plot(K, MSE_U, 'red', label='Uniform')
plt.plot(K, MSE_D, 'k', label='Distance')

plt.ylabel('Mean Square Error')
plt.xlabel('Values of "k"')
plt.legend()
plt.title("The Mean Square Error against Nearest Neighbours")
plt.show()
print(len(K))
   

"""--------------------------------------------------------------"""

"""Linear Regression"""

LR = LinearRegression()


LR.fit(X_train, Y_train)

#pre_train = LR.predict(X_train)

#Prediction
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

AccPerc (Y_test,pre_test)

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

# calculating precision score of the algorithm
precision_score(Y_test, Y_.round(), average=None)

print(classification_report(y, relabel))

""" Statistics """
# https://en.wikipedia.org/wiki/Precision_and_recall
#Function to calculate True positives and False positives and True negatives and False negatives

def StatsCalc(TestingData , PredictionData):
    # Converting numpyndarray array to pandas.dataframe to make it callable
    callable = pd.DataFrame(PredictionData)
    
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
    
    for i in range (TestingData.size):
        # Calculating positive conditions
        if TestingData.iloc[i]['athome'] > 0:
            CP = CP + 1
        # Calculating negative conditions
        elif TestingData.iloc[i]['athome'] == 0:
            CN = CN + 1
            
            
        # Calculating true positives
        if TestingData.iloc[i]['athome'] == ((callable.iloc[i][0]).round()) and (TestingData.iloc[i]['athome']) > 0:
            TP = TP + 1
        # Calculating false positives
        elif (TestingData.iloc[i]['athome'] == 0) and ((callable.iloc[i][0]).round() > 0):
            FP = FP + 1
        # calculating true negatives
        elif (TestingData.iloc[i]['athome']) == ((callable.iloc[i][0]).round()) == 0:
            TN = TN + 1
        # Calculating false negatives
        elif ((TestingData.iloc[i]['athome']) > 0) and ((callable.iloc[i][0]).round() == 0):
            FN = FN + 1
        
    print("True positives: " , TP)
    print("False positives: " , FP)
    print("True negitives: " , TN)
    print("False negitives: " , FN)
    print("Conditions positives: " , CP)
    print("Condition negitives: " , CN)
    print("TOTAL: ", TP+FP+TN+FN)
    print("Total Predicted Instances: " , TestingData.size)
    
# Pinting first 20 values of the testing and predicted data
print("This is the testing data: ",Y_test.head(20))   
print("This is the predicted data: ",callable.head(10))

""" Statistics """

#Trying to figure out how to access the data sets
print(Y_test.iloc[2]['athome'])
print(callable.iloc[3][0])


"""------------------------------------------------------"""
"""k-NN Classification"""
ne = KNeighborsClassifier(n_neighbors=3)

Xnt = t_X[['kwh','unit']]

ynt = t_X[['athome']]
 
ne.fit(X_train, Y_train)

predictD = ne.predict(X_test)

new_Pre = pd.DataFrame(predictD)
print(Y_test.head(10)['athome'], new_Pre.head(10)[0])

AccPerc(Y_test, predictD)
StatsCalc(Y_test, predictD)

print(Y_test.head(20))
print(Y_train.head(20))
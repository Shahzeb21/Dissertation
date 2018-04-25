# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 19:07:51 2018

@author: shahz
"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import sklearn
import pandas as pd
from sklearn.metrics import precision_score
from matplotlib.ticker import FuncFormatter


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
#http://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html#sphx-glr-auto-examples-linear-model-plot-ols-py

t_X = pd.read_csv(r'''C:\Users\shahz\Documents\303 - RESEARCH 2018\Dissertation\Research\electricitydatasetUnit.csv''')

X = t_X[['kwh','unit']]

y = t_X[['athome']]

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.33, random_state = 4)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(X_train, Y_train)

# Make predictions using the testing set
y_pred = regr.predict(X_test)

print(X_test.size)
print(Y_test.size)

# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(Y_test, y_pred))

# Plot outputs
plt.scatter(X_test.mean(axis=1), Y_test,  color='black')
plt.plot(X_test,y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()

plt.scatter(X_test.mean(axis=1), Y_test, c='k', label='data')
plt.scatter(X_test.mean(axis=1), y_pred, c='lightbrown', label='prediction')
plt.axis(ymin=-0.5, ymax=2.5)
plt.legend(loc='best')
plt.title("Linear Regression - Actual VS Predicted data")
plt.show

AccPercLinear(Y_test , y_pred)
StatsCalc(Y_test , y_pred)

x = np.arange(4)
money = [541e5, 214e5, 145e5, 102e5]


def statistics(x, pos):
    'The two args are the value and tick position'
    print('%1f' % (x * 1000e-6))
    return '%1f' % (x * 1000e-6)

print(statistics)

formatter = FuncFormatter(statistics)
print(formatter)
fig, ax = plt.subplots()
ax.yaxis.set_major_formatter(formatter)
plt.bar(x, money)
plt.xticks(x, ('TP', 'FP', 'TN', 'FN'))
plt.show()
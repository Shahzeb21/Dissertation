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
plt.scatter(X_test.mean(axis=1), y_pred, c='b', label='prediction')
plt.axis(ymin=-0.5, ymax=2.5)
plt.legend()
plt.title("Actual data vs Predicted results")
plt.show

AccPerc(Y_test , y_pred)

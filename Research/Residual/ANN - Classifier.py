# -*- coding: utf-8 -*-
"""
Created on Fri Apr  6 18:48:29 2018

@author: shahz
"""

from sklearn.neural_network import MLPClassifier
import pandas as pd
import sklearn

t_X = pd.read_csv(r'''C:\Users\shahz\Documents\303 - RESEARCH 2018\Dissertation\Research\electricitydatasetUnit.csv''')
X = t_X[['kwh','unit']]

y = t_X[['athome']]

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.33, random_state = 4)

clf = MLPClassifier(activation='logistic', alpha=1e-05,
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='adaptive',
       learning_rate_init=0.001, batch_size=32, max_iter=2000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)

clf.fit(X, y)  
"""                       
MLPClassifier(activation='logistic', alpha=1e-05,
       beta_1=0.9, beta_2=0.999, early_stopping=False,
       epsilon=1e-08, hidden_layer_sizes=(5, 2), learning_rate='adaptive',
       learning_rate_init=0.010, batch_size=32, max_iter=2000, momentum=0.9,
       nesterovs_momentum=True, power_t=0.5, random_state=1, shuffle=True,
       solver='sgd', tol=0.0001, validation_fraction=0.1, verbose=False,
       warm_start=False)
"""
clf.predict(X_test)

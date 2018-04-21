# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 00:26:59 2018

@author: shahz
"""

import keras
import sklearn
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

# Generate dummy data
import numpy as np
t_X = pd.read_csv(r'''C:\Users\shahz\Documents\303 - RESEARCH 2018\Dissertation\Research\electricitydatasetUnit.csv''')
X = t_X[['kwh','unit']]

y = t_X[['athome']]

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.33, random_state = 4)

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 2-dimensional vectors.
model.add(Dense(64, activation='relu', input_dim=2))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, Y_train,
          epochs=50,
          batch_size=None)

#playing around
prediction = model.predict(X_train, batch_size=None, verbose=0, steps=None)
model.predict_on_batch(X_train)

model.evaluate(X_test, Y_test, batch_size=32)
score = model.evaluate(X_test, Y_test, batch_size=32)

print('HELLOO IM THE SCORE: ' , score)
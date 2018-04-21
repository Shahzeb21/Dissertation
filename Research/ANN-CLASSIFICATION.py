#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 15:19:39 2018

@author: huzmorgoth
"""

from keras.models import Sequential
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
import sklearn
from sklearn import model_selection
import pandas as pd
import numpy as np

# Read data
X_t = pd.read_csv(r'''C:\Users\shahz\Documents\303 - RESEARCH 2018\Dissertation\Research\electricitydatasetUnit.csv''')

X = X_t[['kwh','unit']]
y = X_t[['athome']]

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.33, random_state = 4)

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

#labels = train.ix[:,0].values.astype('int32')
#X_train = (train.ix[:,1:].values).astype('float32')
#X_test = (pd.read_csv('../input/test.csv').values).astype('float32')

# convert list of labels to binary class matrix
Y_train = np_utils.to_categorical(y_train) 

# pre-processing: divide by max and substract mean
scale = np.max(X_train)
X_train /= scale
X_test /= scale

mean = np.std(X_train)
X_train -= mean
X_test -= mean

input_dim = X_train.shape[1]
nb_classes = Y_train.shape[1]

# Here's a Deep MLP (DDMLP)
model = Sequential()
model.add(Dense(128, input_dim=input_dim))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.15))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# we'll use categorical xent for the loss, and RMSprop as the optimizer
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print("Training...")
model.fit(X_train, Y_train, nb_epoch=100, batch_size=32, validation_split=0.1, verbose=2)

print("Generating test predictions...")
preds = model.predict_classes(X_test, verbose=0)

print(y_test,"-",preds)

#Predictions = pd.DataFrame(preds) 
#for i in range (y_test.size):
#    print(y_test.iloc[i]['athome'], ' - ' , Predictions.iloc[i][0])
    
AccPerc(y_test,preds)
StatsCalc(y_test,preds)
#def write_preds(preds, fname):
 #   pd.DataFrame({"unit and kwh": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)

#write_preds(preds, "keras-mlp.csv")
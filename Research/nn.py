import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# sigmoid function

def sigmo(x, deriv=False):
    if(deriv==True):
        return (x*(1-x))
        
    return 1/(1+np.exp(-x))
    
# input data
t_X = pd.read_csv(r'''C:\Users\shahz\Documents\Research\electricitydataset.csv''').head(10)

X = t_X.kwh

# actual output

y = t_X.athome
    
# seed

np.random.seed(1)

# synapses (weights)

syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

# training 

for j in range(60000):
    
    # layers
    l0 = X
    l1 = sigmo(np.dot(l0,syn0))
    l2 = sigmo(np.dot(l1,syn1))
    
    # backpropogation 
    l2_err = y - l2
    if (j%10000) == 0:
        print("Error: ", str(np.mean(np.abs(l2_err))))
        
    # compute deltas
    l2_delta = l2_err * sigmo(l2, deriv=True)    
    l1_err = l2_delta.dot(syn1.T)
    l1_delta = l1_err * sigmo(l1, deriv=True)
    
    #update synapses (weights)
    syn0 += l0.T.dot(l1_delta)
    syn1 += l1.T.dot(l2_delta)
    
print("print output after training: ")
print(l2)
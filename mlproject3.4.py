import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

import os



import tensorflow as tf
#from tensorflow import keras


os.chdir(r'I:\Meu Drive\engEletrica\IEEE\3periodo')

dbpath = os.getcwd()+'\\creditcard.csv'
data = pd.read_csv(dbpath)


#Preprocessing


#Additionally, since we are going to train the neural network
#using Gradient Descent, we must scale the input features.

#desconsiderar coluna "Time"
cols = list(data.columns)
cols.remove("Time")
data = data[cols]


#separar train_set, valid_test e  test_set
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=42)

pclass = "Class"

featureNames = list(data.columns)
featureNames.remove(pclass)

train_set = train_data[featureNames]
train_labels = train_data[pclass]

valid_set = valid_data[featureNames]
valid_labels = valid_data[pclass]

test_set = test_data[featureNames]
test_labels = test_data[pclass]



######################

def af(x): #activation function: sigmoid
    return 1/(1+np.exp(-x))

def daf(x): #derivative af
    return np.exp(-x)/(1+np.exp(-x))**2





nFeatures = train_set.shape[1]
hidden_layer_size = (100)

mBatchSize = 35
epochs = 5
learnRate = 0.5


coef0 = np.random.rand(hidden_layer_size,nFeatures)# * 100
coef1 = np.random.rand(1,hidden_layer_size)# * 100

biasw0 = np.random.rand(hidden_layer_size,1)# * 100
biasw1 = np.random.rand(1,1)# * 100





for i in range(epochs):
    #count = 0
    #instans = list(range(20))
    nInstances = train_set.shape[0]
    instPerEpoch = list(range(0,nInstances,mBatchSize))
    instPerEpoch.append(nInstances)
    
    for i in range(len(instPerEpoch)-1):
        #print(train_set[instPerEpoch[i]:instPerEpoch[i+1]])
        x = train_set[instPerEpoch[i]:instPerEpoch[i+1]] #miniBatch
        y = train_labels[instPerEpoch[i]:instPerEpoch[i+1]]
        
        x = x.to_numpy()
        #y = y.to_numpy().reshape(-1,1)
        
        #feedforward
        np.matmul(coef0, x.T) + biasw0
        res0 = af(np.matmul(coef0, x.T) + biasw0)
        res1 = af(np.matmul(coef1,res0) + biasw1)
        
        test_pred2 = np.rint(res1)
        cm = confusion_matrix(test_pred2[0], y.to_numpy())
        print(cm)
        #medir acurácia...
        
        
        
        #cost function
        y_train = y.to_numpy().reshape(-1,1)
        cf = (res1 - y_train)**2
        np.rint(cf)
        
        
    
        #backprop
        grad_coef0 = np.zeros(coef0.shape)
        grad_coef1 = np.zeros(coef1.shape)
        grad_biasw0 = np.zeros(biasw0.shape)
        grad_biasw1 = np.zeros(biasw1.shape)
        
        
        #segunda camada
        dc_daL1 = 2*(res1-y_train.T) #cost_derivative
        zL1 = np.matmul(coef1,res0) + biasw1
        daL1_dzL1 = daf(zL1)
        dzL1_dwL1 = res0
        
        grad_coef1 = (dc_daL1*daL1_dzL1)*dzL1_dwL1
        grad_coef1 = np.mean(grad_coef1,axis=1).reshape(-1,1).T
        
        grad_biasw1 = np.mean(dc_daL1*daL1_dzL1,axis=1)
        
        
        ###primeira camada
        dzL1_daL0 = coef1
        zL0 = np.matmul(coef0,x.T) + biasw0
        da0_dz0 = daf(zL0)
        dz0_dw0 = x
        
        
        dfullcost = np.mean(np.matmul(dzL1_daL0.T,dc_daL1*daL1_dzL1),axis=1).reshape(-1,1)
        grad_coef0 = np.matmul(da0_dz0,x)*dfullcost
        
        grad_biasw0 = np.mean(da0_dz0*dfullcost,axis=1).reshape(-1,1)
        #grad_biasw0 = np.mean(grad_coef1*da0_dz0, axis = 1).reshape(-1,1)
        
        #atualização dos valores
        
        coef0 = coef0 - learnRate*grad_coef0
        coef1 = coef1 - learnRate*grad_coef1
        
        biasw0 = biasw0 - learnRate*grad_biasw0
        biasw1 = biasw1 - learnRate*grad_biasw1
      
        



import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

import os

import math

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
learnRate = 0.005


#coef0 = np.random.rand(hidden_layer_size,nFeatures)# * 100
#coef1 = np.random.rand(1,hidden_layer_size)# * 100

#biasw0 = np.random.rand(hidden_layer_size,1)# * 100
#biasw1 = np.random.rand(1,1)# * 100

coef0 = np.random.randn(hidden_layer_size,nFeatures)# * 100
coef1 = np.random.randn(1,hidden_layer_size)# * 100

biasw0 = np.random.randn(hidden_layer_size,1)# * 100
biasw1 = np.random.randn(1,1)# * 100



for i in range(epochs):
    #count = 0
    #instans = list(range(20))
    nInstances = train_set.shape[0]
    instPerEpoch = list(range(0,nInstances,mBatchSize))
    instPerEpoch.append(nInstances)
    
    for k in range(len(instPerEpoch)-1):
        #print(train_set[instPerEpoch[i]:instPerEpoch[i+1]])
        x = train_set[instPerEpoch[i]:instPerEpoch[i+1]] #miniBatch
        y = train_labels[instPerEpoch[i]:instPerEpoch[i+1]]
        
        x = x.to_numpy().T
        y = y.to_numpy().T
        #y = y.to_numpy().reshape(-1,1)
        
        #x = train_set.to_numpy().T
        #y = train_labels.to_numpy().T
        # Feedforward
        activation = x
        
        #armazenar ativações
        activations = [x]
        
        #armazenar vetores z, capada por camada
        zs = []
        
        
        z0 = np.dot(coef0,activation)+biasw0
        activation0 = af(z0)
        activations.append(activation0)
        
        
        z1 = np.dot(coef1,activation0)+biasw1
        activation1 = af(z1)
        activations.append(activation1)
        
        
        #backward pass
        delta1 = 2*(activation1-y) * daf(z1)
        grad_biasw1 = delta1  #nabla_b[-1] 
        grad_coef1 = np.dot(delta1,activation0.transpose()) #nabla_w[-1]
        
        
        sp0 = daf(z0)
        delta0 = np.dot(coef1.transpose(), delta1) * sp0
        grad_biasw0 = delta0
        grad_coef0 = np.dot(delta0,x.T)
        
        #if math.isnan(np.mean(grad_coef0)):
        #    print('NAN')
        #    a = input()
        #else:
        #    print()
        #    print(np.mean(grad_coef0))
        #    print(np.mean(grad_coef1))
            
        
        coef0 = coef0 - learnRate*grad_coef0
        coef1 = coef1 - learnRate*grad_coef1
        
        biasw0 = biasw0 - learnRate*grad_biasw0
        biasw1 = biasw1 - learnRate*grad_biasw1
        


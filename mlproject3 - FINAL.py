import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import os
import math
import tensorflow as tf



#funções a serem utilizadas
def accuracy(confusion_matrix):
   diagonal_sum = confusion_matrix.trace()
   sum_of_all_elements = confusion_matrix.sum()
   return diagonal_sum / sum_of_all_elements

####### Funções de Ativação ###############
def relU(x):
    return np.maximum(0,x)
def drelU(x):
    return (x > 0) * 1
     
    
def sigmoid(x): #activation function: sigmoid
    return 1/(1+np.exp(-x))

def dsigmoid(x): #derivative af
    return np.divide(np.exp(-x),(1+np.exp(-x))**2)


def af(x):
    return sigmoid(x)
def daf(x):
    return drelU(x)


os.chdir(r'I:\Meu Drive\engEletrica\IEEE\3periodo')

dbpath = os.getcwd()+'\\creditcard.csv'
data = pd.read_csv(dbpath)


#Preprocessing
col_names = ['Amount']
features = data[col_names]
scaler = StandardScaler().fit(data[col_names])
data[col_names] = pd.DataFrame(scaler.transform(features), columns = col_names)


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



nFeatures = train_set.shape[1]
hidden_layer_size = (100)

mBatchSize = 300
epochs = 3
learnRate = 0.05



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
    
    """
    #overall loss no validation set:
    print('epoch:', i)
    z0 = np.dot(coef0,valid_set.to_numpy().T)+biasw0
    a0 = af(z0)
    z1 = np.dot(coef1,a0)+biasw1
    a1 = af(z1)
    loss = np.mean((a1-valid_labels.to_numpy().T.reshape(a1.shape))**2)
    print('overall loss: ', loss)
    """
    for k in range(len(instPerEpoch)-1):
        
        #overall loss no validation set
        if k%1000 == 0:
            print('epoch/mBatch: ', i,'-',k)
            z0 = np.dot(coef0,valid_set.to_numpy().T)+biasw0
            a0 = af(z0)
            z1 = np.dot(coef1,a0)+biasw1
            a1 = af(z1)
            loss = np.mean((a1-valid_labels.to_numpy().T.reshape(a1.shape))**2)
            print('overall loss: ', loss)
        
        #print(f'epoch({i}) mBatch({k}):')
        #print(train_set[instPerEpoch[i]:instPerEpoch[i+1]])
        x = train_set[instPerEpoch[i]:instPerEpoch[i+1]] #miniBatch
        y = train_labels[instPerEpoch[i]:instPerEpoch[i+1]]
        
        x = x.to_numpy().T
        y = y.to_numpy().reshape(-1,1).T
        
        bsize = x.shape[1]
        
        # Feedforward
        z0 = np.dot(coef0,x)+biasw0
        activation0 = af(z0)
        
        z1 = np.dot(coef1,activation0)+biasw1
        activation1 = af(z1)
        
        
        #Backward pass
        delta1 = 2*(activation1-y) * daf(z1)
        grad_biasw1 = delta1  #nabla_b[-1] 
        grad_coef1 = np.dot(delta1,activation0.transpose())/bsize #nabla_w[-1]
        
        
        sp0 = daf(z0)
        delta0 = np.dot(coef1.transpose(), delta1) * sp0
        grad_biasw0 = delta0
        grad_coef0 = np.dot(delta0,x.T)/bsize
        
        
        coef0 = coef0 - learnRate*grad_coef0
        coef1 = coef1 - learnRate*grad_coef1
        
        biasw0 = biasw0 - learnRate*np.mean(grad_biasw0,axis=1).reshape(biasw0.shape)
        biasw1 = biasw1 - learnRate*np.mean(grad_biasw1,axis=1).reshape(biasw1.shape)
        

    #Acurácia nos dados de teste
    #selecionar dados:
    x = train_set.to_numpy()
    y = train_labels.to_numpy()
    
    z0 = np.dot(coef0,x.T)+biasw0
    a0 = af(z0)
    z1 = np.dot(coef1,a0)+biasw1
    a1 = af(z1)
    test_pred = np.round(a1)
    test_pred = (test_pred > 1)*2
    
    #Comparing the predictions against the actual observations in y_val
    cm = confusion_matrix(test_pred[0], y.reshape(1,-1)[0])
    print(cm)


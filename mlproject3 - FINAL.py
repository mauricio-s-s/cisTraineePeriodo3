import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTE
import os
import math
import tensorflow as tf



#funções a serem utilizadas
def accuracy(cm):
   diagonal_sum = cm.trace()
   sum_of_all_elements = cm.sum()
   
   gen_acc = diagonal_sum / sum_of_all_elements
   
   if cm.shape == (2,2):
       tp = cm[0,0]
       fp = cm[0,1]
       tn = cm[1,1]
       fn = cm[1,0]
       
       print()
   return gen_acc

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
data.dtypes


#desconsiderar coluna "Time"
data = data.drop("Time",axis = 'columns')


#aplicar StandardScaler nas features de entrada
#col_names = ['Amount']
cols = list(data.columns)
cols.remove('Class')
features = data[cols]
scaler = StandardScaler().fit(data[cols])
data[cols] = pd.DataFrame(scaler.transform(features), columns = cols)


X = data.drop('Class',axis = 'columns')
y = data['Class']
y.value_counts()
""" 
0    284315
1       492
"""

#corrigir undersampling
#Método SMOTE
smote = SMOTE(sampling_strategy='minority')
Xadj, yadj = smote.fit_resample(X,y)
yadj.value_counts()


""" 
0    284315
1    284315
"""


#separar train_set, valid_test e  test_set
X_train, X_test, y_train, y_test = train_test_split(Xadj, yadj, test_size=0.2, random_state=42, stratify=yadj)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)



nFeatures = X_train.shape[1]
hidden_layer_size = (100)

mBatchSize = 40
epochs = 10
learnRate = 0.5


coef0 = np.random.randn(hidden_layer_size,nFeatures)
coef1 = np.random.randn(1,hidden_layer_size)#

biasw0 = np.random.randn(hidden_layer_size,1)
biasw1 = np.random.randn(1,1)

for i in range(epochs):
    #count = 0
    #instans = list(range(20))
    nInstances = X_train.shape[0]
    instPerEpoch = list(range(0,nInstances,mBatchSize))
    instPerEpoch.append(nInstances)
    

    for k in range(len(instPerEpoch)-1):
        x = X_train[instPerEpoch[i]:instPerEpoch[i+1]] #miniBatch
        y = y_train[instPerEpoch[i]:instPerEpoch[i+1]]
        
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
        
    print()
    print('epoch: ', i)
    z0 = np.dot(coef0,X_valid.to_numpy().T)+biasw0
    a0 = af(z0)
    z1 = np.dot(coef1,a0)+biasw1
    a1 = af(z1)
    loss = np.mean((a1-y_valid.to_numpy().T.reshape(a1.shape))**2)
    print('overall loss: ', loss)
    
    #Acurácia nos dados de teste
    #selecionar dados:
    x = X_test.to_numpy()
    y = y_test.to_numpy()
    
    z0 = np.dot(coef0,x.T)+biasw0
    a0 = af(z0)
    z1 = np.dot(coef1,a0)+biasw1
    a1 = af(z1)
    test_pred = np.round(a1)
    #test_pred = (test_pred > 1)*2
    
    #Comparing the predictions against the actual observations in y_val
    cm = confusion_matrix(test_pred[0], y.reshape(1,-1)[0])
    print(cm)
    print("Classification Report: \n", classification_report(y.reshape(1,-1)[0], test_pred[0]))

    


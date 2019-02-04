# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 13:31:34 2019

@author: Hubert Kyeremateng-B
"""
import numpy as np
#from ARPSimulator import ARPSimulator as arp
import pandas as pd
from ARPSimulator import ARPSimulator as arp
class CustomSVR:
    def __init__(self, lr=0.01, num_iter=100000, fit_intercept=True, verbose=False):
        self.lr = lr
        self.num_iter = num_iter
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def __add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)
    
    def fit(self, X, y):
        if self.fit_intercept:
            X = self.__add_intercept(X)
        # weights initialization
        self.theta = np.zeros(X.shape[1])
        for i in range(self.num_iter):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)
            gradient = np.dot(X.T, (h - y)) / y.size
            self.theta -= self.lr * gradient
        
        if(self.verbose == True and i % 10000 == 0):
            z = np.dot(X, self.theta)
            h = self.__sigmoid(z)

    def __sigmoid(self,x):
        x = x.astype(np.float32)
        return 1 / (1 + np.exp(-x))
    
    def __loss_function(self,h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def __predict_probs(self,X):
        return self.__sigmoid(np.dot(X))
    
    def predict(self,X, threshold=0.5):
        return self.__predict_probs(X) >= threshold
    
svr = CustomSVR()

N = 500
datas = arp.generateFreqEnergy(arp,0.8,0.8,N)
powerData = pd.DataFrame(np.array(datas).reshape(-1,1),datas)
powerData = powerData[powerData.columns[0:]].values
powerLvlLambda = powerData.transpose();
dLen = 100 #length of the energy detector
N_train = (dLen*N)//2-dLen+1; #training data length - 14901
#Training label ground truth/target
wLen = 5*dLen; 
N_test = (dLen*N)//2; #test data length - 15000

N = N_train+N_test; #total data length - 29901

sample_len = 5

#input window length
trainLbl = np.zeros((1,N_train-wLen));

r = 0
for i in np.arange(wLen,N_train-1):
    trainLbl.itemset(r,powerLvlLambda[0,i]);
    r = r +1;
    

#Traing and test input data
trainData = np.zeros((wLen,N_train-wLen));
testData = np.zeros((wLen,N_test-wLen));
###### totalAvgPwr is one-dimensional array#####
###### trainData in a multi-dimensional array######

for s in np.arange(r):
    for k in  np.arange(wLen-1):
        trainData.itemset((k,s),np.real(powerLvlLambda[0,s+k]))

a = N_test


for b in np.arange(a-wLen):
    for t in  np.arange(wLen):
        testData.itemset((t,b),np.real(powerLvlLambda[0,N_train+t+b]));


label_reshape = trainLbl.reshape((N_train-wLen))
train_reshape = trainData.transpose()

svr.fit(train_reshape,label_reshape)
svr.predict(testData)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 00:17:53 2018

@author: hkyeremateng-boateng
"""

import numpy as np
import matplotlib.pyplot as plt
import time as tic
from sklearn.svm import LinearSVR
import pandas as pd
#from sklearn.model_selection import train_test_split

N = 300
dLen = 100 #length of the energy detector
N_train = dLen*N//2-dLen+1; #training data length - 14901
#Training label ground truth/target
wLen = 5*dLen; 
N_test = dLen*N//2; #test data length - 15000
N = N_train+N_test; #total data length - 29901
sample_len = 5
data = pd.read_csv('svmDataSet.csv', header=None,sep='\n')
data = data[data.columns[0:]].values
datas = data.transpose()

print(tic.clock())
#input window length
trainLbl = np.zeros((1,N_train-wLen));

r = 0
for i in np.arange(wLen,N_train-1):
    trainLbl.itemset(r,datas[0,i]);
    r = r +1;
    
#Traing and test input data
trainData = np.zeros((wLen,N_train-wLen));
testData = np.zeros((wLen,N_test-wLen));
###### totalAvgPwr is one-dimensional array#####
###### trainData in a multi-dimensional array######

for s in np.arange(r):
    for k in  np.arange(wLen-1):
        trainData.itemset((k,s),np.real(datas[0,s+k]))

a = N_test-1
c = wLen-1
for b in np.arange(a-wLen):
    print(b)
    for t in  np.arange(wLen-1):
        testData[t,b] = np.real(datas[0,N_train+t]);
        
predAcc_obs2_coh = np.zeros((dLen,1));
theSVR = LinearSVR()
print("Starting to fit model")
print(tic.clock())

label_reshape = trainLbl.reshape((N_train-wLen))
train_reshape = trainData.conj().transpose()
testData_reshape = testData.conj().transpose()
clf = theSVR.fit(train_reshape,label_reshape)

fSteps = dLen; #tracks number of future steps to predict
score = np.zeros((0,N_test));
predicted = np.zeros((fSteps, N_test-wLen));

nData = testData_reshape;
for i in np.arange(0,sample_len):
    predicted[i] = clf.predict(testData.conj().transpose()).conj().transpose();
    nData = np.concatenate((testData[1:wLen:,],predicted[i:sample_len]));


predSet = np.zeros((fSteps, N_test-wLen));
setCnt = 0;
obsSample = np.zeros((1,N_test-wLen));
for i in np.arange(0,N_test-wLen):
    predSet[setCnt,i-setCnt:i-setCnt+fSteps] = predicted[:,i].conj().transpose()
    if (setCnt+1)==fSteps:
        #obsSample[i-setCnt] = 1;
        setCnt = 0;
    else:
        setCnt = setCnt + 1;
    
    
score = predicted[0:sample_len];
fig, ax = plt.subplots()
plt.plot(np.array([i for i in np.arange(N_test-wLen)]),10*np.log10(np.abs(np.real(datas[wLen:N])))-30,np.array([i for i in np.arange(N_test-wLen)]),10*np.log10(abs(score))-30)
#plt.plot(0:N_test-wLen,10*np.log10(np.abs(score))-30)
'''
10*log10(abs(real(totalAvgPwr(N_train+1+wLen:N))))-30,...
    1:N_test-wLen,10*log10(abs(score))-30
'''
plt.title('one-step ahead prediction')
#plt.legend('Input Signal','Prediction')
plt.xlabel('Samples')
plt.ylabel('Magnitude (dBm)')
plt.show()
print(tic.clock())
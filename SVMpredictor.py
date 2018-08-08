#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 00:17:53 2018

@author: hkyeremateng-boateng
"""

import numpy as np
#import matplotlib.pyplot as plt
import time as tic
from sklearn.svm import SVR, SVC
import pandas as pd
from sklearn.model_selection import train_test_split

N = 300
dLen = 100 #length of the energy detector
N_train = dLen*N//2-dLen+1; #training data length - 149901
#Training label ground truth/target
wLen = 5*dLen; 
N_test = dLen*N//2; #test data length - 150000
N = N_train+N_test; #total data length - 299901
sample_len = 5
data = pd.read_csv('arp_dataset.txt',sep='\n')
data = data[data.columns[0:1]].values
datas = np.transpose(data)

print(tic.time())
#input window length
trainLbl = np.zeros((1,N_train-1));

for i in np.arange(N_train-wLen):
    trainLbl.itemset(i,datas[0,i]); 
    
#Traing and test input data
trainData = np.zeros((wLen,N_train-1));
testData = np.zeros((wLen,N_test-wLen-1));
###### totalAvgPwr is one-dimensional array#####
###### trainData in a multi-dimensional array######

for s in np.arange(N_train):
    for k in  np.arange(wLen):
        trainData.itemset((k,s-wLen+1),np.abs(datas[0,s+k]))

for b in np.arange(N_test):
    for t in  np.arange(wLen):
        testData[t,b-wLen-1] = np.abs(datas[0,b+t-1]);
        
predAcc_obs2_coh = np.zeros((dLen,1));
theSVR = SVR(kernel='linear',max_iter=10)
print("Starting to fit model")


label_reshape = trainLbl.reshape((N_train-1))
train_reshape = trainData.transpose()
testData_reshape = testData.transpose()
clf = theSVR.fit(train_reshape,label_reshape)

fSteps = dLen; #tracks number of future steps to predict
score = np.zeros((1,N_test))
predicted = np.zeros((fSteps, N_test-wLen-1));

nData = testData_reshape;
for i in np.arange(0,sample_len):
    predicted[i] = clf.predict(nData);
    nData = np.concatenate((testData[0:wLen],predicted[i:sample_len])).reshape((wLen,-1))
    #for r in np.arange(0,i):
        #nData = [predicted[r]]

for i in np.arange(sample_len):
    score = predicted[i];
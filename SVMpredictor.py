#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 00:17:53 2018

@author: hkyeremateng-boateng
"""

import numpy as np
#import matplotlib.pyplot as plt
import time as tic
from sklearn import svm
import pandas as pd

N = 300
dLen = 100 #length of the energy detector
N_train = dLen*N//2-dLen+1; #training data length - 149901
#Training label ground truth/target
wLen = 5*dLen; 
N_test = dLen*N//2; #test data length - 150000
N = N_train+N_test; #total data length - 299901

data = pd.read_csv('arp_dataset.txt',sep='\n')
data = data[data.columns[0:1]].values
datas = np.transpose(data)

print(tic.time())
#input window length
trainLbl = np.zeros((1,N_train-wLen));
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
theSVR = svm.SVR(kernel='linear')
"""
Error message - 
ValueError: Expected 2D array, got 1D array instead:
array=[1.02042227e-05 1.01761438e-05 1.02209320e-05 ... 1.00228807e-05
 1.00395029e-05 1.00579164e-05].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
"""

test= theSVR.fit(trainData.flatten(),trainLbl.flatten())

fSteps = dLen; #tracks number of future steps to predict
predicted = np.zeros((fSteps, N_test-wLen));

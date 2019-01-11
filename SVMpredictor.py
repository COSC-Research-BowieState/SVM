#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 00:17:53 2018

@author: hkyeremateng-boateng
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import sklearn.svm as svm
import pandas as pd
from datetime import datetime
import sklearn.metrics as metrics
import ARPSimulator as arp

def generateFreqData(lambda1,lambda2,numberOfSamples):
        totalPwrLvl = arp.ARPSimulator.generateFreqEnergy(arp,lambda1,lambda1,numberOfSamples)
        powerData = pd.DataFrame(np.array(totalPwrLvl).reshape(-1,1),totalPwrLvl)
        powerData = powerData[powerData.columns[0:]].values
        powerLvlLambda = powerData.transpose();
        return powerLvlLambda;
        
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

lambda1=0.8
lambda2=0.8
N = 3000
dLen = 100 #length of the energy detector
N_train = dLen*N//2-dLen+1; #training data length - 14901

#Training label ground truth/target
wLen = 5*dLen; 
N_test = (dLen*N//2); #test data length - 15000
N = N_train+N_test; #total data length - 29901
sample_len = 5

datas = generateFreqData(lambda1,lambda2,N)
print(datetime.now())
#input window length
trainLbl = np.zeros((1,N_train-wLen));

r = 0
for i in np.arange(wLen,N_train-1):
    trainLbl.itemset(r,datas[0,i]);
    r = r +1;
    
#Training and test input data
trainData = np.zeros((wLen,N_train-wLen));
testData = np.zeros((wLen,N_test-wLen));
trailData = np.zeros((wLen,N_test-wLen));
totalPwr_score = np.zeros((N_test-wLen));
prediction_score = np.zeros((N_test-wLen));
accuracy_percent = np.zeros((N_test-wLen));
###### totalAvgPwr is one-dimensional array#####
###### trainData in a multi-dimensional array######

for s in np.arange(r):
    for k in  np.arange(wLen-1):
        trainData.itemset((k,s),np.real(datas[0,s+k]))

a = N_test


for b in np.arange(a-wLen):
    for t in  np.arange(wLen):
        testData.itemset((t,b),np.real(datas[0,N_train+t+b]));

predAcc_obs2_coh = np.zeros((dLen,1));

print("Starting to fit model")
print(datetime.now())

label_reshape = trainLbl.reshape((N_train-wLen))
train_reshape = trainData.transpose()
testData_reshape = testData.transpose()

clf = svm.LinearSVR(epsilon=10e-20, C=10e20, max_iter=900000, dual=True,random_state=0,tol=10e-6,verbose=6,loss='squared_epsilon_insensitive').fit(train_reshape,label_reshape)

fSteps = dLen; #tracks number of future steps to predict
score = np.zeros((1,N_test));
predicted = np.zeros((fSteps, N_test-wLen));
print(clf)
nData = testData;

for i in range(0,sample_len):
    predicted[i] = clf.predict(nData.transpose());
    nData = np.concatenate((testData[1:wLen:,],predicted[i:i+1,:]));


predSet = np.zeros((fSteps, N_test-wLen));
setCnt = 0;
obsSample = np.zeros((1,N_test-wLen));

for i in np.arange(0,N_test-wLen):
    predSet[setCnt,i-setCnt:i-setCnt+fSteps] = predicted[:,i].transpose()
    if (setCnt+1)==fSteps:
        #obsSample[i-setCnt] = 1;
        setCnt = 0;
    else:
        setCnt = setCnt + 1;

score = predicted[1,:]

''' Retrieve Total Power and Prediction scores'''

totalPwr_score = 10*np.log10(np.abs(np.real(datas[0,N_train+wLen:N])))-30;
prediction_score = 10*np.log10(np.abs(score))-30
accuracy_percent = ((np.abs(totalPwr_score) - np.abs(prediction_score)) + np.abs(prediction_score))/100 # Accuracy rate as a percent 

test_01 = np.zeros((10))
score_trans = test_01.transpose()  
subplot_range = np.array([i for i in np.arange(N_test-wLen)]);
fig = plt.figure(figsize=(30,30))
plt.subplot(2,1,1)
plt.plot(subplot_range,totalPwr_score,subplot_range,prediction_score)

plt.title('one-step ahead prediction')
plt.legend(['Input Signal','Prediction'])
plt.xlabel('Samples')
plt.ylabel('Magnitude (dBm)')


plt.savefig("3000samples_0.8lambda900k_iterations.png")
plt.show()
print(datetime.now())


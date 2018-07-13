#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:41:56 2018

@author: hubert kyeremateng-boateng
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 22:44:40 2018

@author: hkyeremateng-boateng
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy import special
import pandas as pd

N = 300
dLen = 10 #length of the energy detector
N_train = dLen*N//2-dLen+1; #training data length - 149901
#Training label ground truth/target
wLen = 5*dLen; 
N_test = dLen*N//2; #test data length - 150000
N = N_train+N_test; #total data length - 299901

data = pd.read_csv('arp_dataset.txt',sep='\n')
data = data[data.columns[0:1]].values
datas = np.transpose(data)


#input window length
trainLbl = np.zeros((1,N_train));
for i in range(N_train):
    trainLbl.itemset(i,datas[0,i]); #Complete
    
#Traing and test input data
trainData = np.zeros((wLen,N_train));
testData = np.zeros((wLen,N_test-wLen));
###### totalAvgPwr is one-dimensional array#####
###### trainData in a multi-dimensional array######

for s in np.arange(wLen,N_train-1):
    for k in  range(wLen):
        trainData.itemset((k,s),datas[0,s-wLen+1])
tests = np.arange(wLen,N_test-1)

for i in np.arange(wLen,N_test-1):
    #Input consists of present state and wLen previous states
    for k in  range(wLen):
        testData.itemset((k,i),datas[0,i-wLen])       

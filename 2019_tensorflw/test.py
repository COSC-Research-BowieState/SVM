# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 14:23:10 2018

@author: Hubert Kyeremateng-B
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:13:11 2018

@author: hkyeremateng-boateng
"""
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
import pandas as pd
from datetime import datetime
from ARPSimulator import ARPSimulator as arp
import csv
from sklearn.metrics import mean_squared_error
from numba import autojit
from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import Dense

rng = np.random.RandomState(42)
class SVMPredictor:
    def calcAccuracy(self, lambda1, lambda2):
        return 0;
    
    #@autojit
    def predictor(self,numberOfSamples, datas):
        
        N = numberOfSamples
        dLen = 100 #length of the energy detector
        N_train = dLen*N//2-dLen+1; #training data length - 14901
        #Training label ground truth/target
        wLen = 5*dLen; 
        N_test = dLen*N//2; #test data length - 15000
        N = N_train+N_test; #total data length - 29901

        sample_len = 5
        
        #input window length
        trainLbl = np.zeros((0,N_train-wLen));

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
        
        a = N_test
        

        for b in np.arange(a-wLen):
            for t in  np.arange(wLen):
                testData.itemset((t,b),np.real(datas[0,N_train+t+b]));

        
        label_reshape = trainLbl.reshape((N_train-wLen))
        train_reshape = trainData.transpose()
        clf = KerasRegressor(build_fn=self.svm_model, nb_epoch=100, batch_size=5,verbose=5)
        clf = clf.fit(train_reshape,label_reshape)
        #clf = self.fit(train_reshape,label_reshape)        
        fSteps = dLen; #tracks number of future steps to predict

        predicted = np.zeros((fSteps, N_test-wLen));
 
        nData = testData;
        for i in np.arange(0,sample_len):
            predicted[i] = clf.predict(nData.transpose());
            nData = np.concatenate((testData[1:wLen:,],predicted[i:i+1,:]));
        
        
        predSet = np.zeros((fSteps, N_test-wLen));
        setCnt = 0;

        
        for i in np.arange(0,N_test-wLen):
            predSet[setCnt,i-setCnt:i-setCnt+fSteps] = predicted[:,i].transpose()
            if (setCnt+1)==fSteps:
                #obsSample[i-setCnt] = 1;
                setCnt = 0;
            else:
                setCnt = setCnt + 1;
        return predicted
    
    def plotAccuracy(self, lambda_1,lambda_2):
        
        return 0;
    def svm_model():
        model = Sequential()
        model.add(Dense(14, input_dim=13, init='normal', activation='relu'))
        model.add(Dense(7, init='normal', activation='relu'))
        model.add(Dense(1, init='normal'))
        model.compile(loss='mean_squared_error', optimizer = 'adam')
        return model        
    #@autojit
    def fit(self,train_reshape,label_reshape):
        clf = svm.LinearSVR(epsilon=10e-90, C=10e5, max_iter=2000,verbose=6, dual=True,random_state=0,loss='squared_epsilon_insensitive',tol=10e-2).fit(train_reshape,label_reshape)
        return clf
    def plotSVM(self, totalPwrLvl, prediction,sampleSize):
        dLen = 100 #length of the energy detector
        N = sampleSize
        N_train = dLen*N//2-dLen+1; #training data length - 14901
        #Training label ground truth/target
        wLen = 5*dLen; 
        N_test = dLen*N//2; #test data length - 15000
        N = N_train+N_test;
        
        score = prediction[1,:]
        
        totalPwr_score = np.zeros((N));
        prediction_score = np.zeros((N_test-wLen));
        
        
        totalPwr_score = 10*np.log10(np.abs(np.real(totalPwrLvl[0,N_train+wLen:N])))-30
        
        prediction_score = 10*np.log10(np.abs(score))-30
        subplot_range = np.array([i for i in np.arange(N_test-wLen)]);
        
        fig = plt.figure(figsize=(30,10))
        
        plt.plot(subplot_range,totalPwr_score,subplot_range,prediction_score)
        plt.title('one-step ahead prediction')
        plt.legend(['Input Signal','Prediction'])
        plt.xlabel('Samples')
        plt.ylabel('Magnitude (dBm)')
        
        fig.savefig("Images/3000Samples_0.9Lambda_900Kiterations.png")
        plt.show()

    def generatereqByLambda(self,lambda1, lambda2,numberOfSamples):
        lambda_range=0.1
        lst = len(np.arange(lambda_range,lambda1,lambda_range)) # Get the length of the lambda range
         
        lambda1_range = np.zeros((lst,lst))
        r = 0;
        plt.figure(figsize=(20,10))
        plt.subplot(1,1,1)
        #cmap = plt.get_cmap('PiYG')
        #norm = BoundaryNorm(levels,ncolors=cmap, clip=True)
        #plt_two.pcolormesh(plt_range,lambda1_range[0,:],plt_range,lambda1_range[0,:],cmap=cmap,norm=norm)
        #axins = zoomed_inset_axes(plt_two, 2.5, loc=2) 
        #plt_range = np.array([i for i in np.arange(lambda_range,lambda1,lambda_range)]);
        for i in np.arange(lambda_range,lambda1,lambda_range):
            s = 0;
            for j in np.arange(lambda_range,lambda2, lambda_range):
                
                powerLvlLambda = svmp.generateFreqData(lambda1,lambda2,numberOfSamples)
                
                prediction = svmp.predictor(numberOfSamples,powerLvlLambda)
                
                score = prediction[1,:]
                accuracy = svmp.calculateAccuracy(score, powerLvlLambda, numberOfSamples)
                lambda1_range.itemset((r,s),accuracy/100)
                colors = np.random.rand(lst)
                print(colors)
                plt.scatter(i,j,c=colors, s=accuracy*15, alpha=0.75,cmap=plt.get_cmap('PiYG'))
                s = s + 1;
            r = r+1;
        plt.colorbar()
        plt.show()
    
    def saveARPData(self, data):
        with open("svmDataSet.csv", 'w') as arp_dataset:
            wr = csv.writer(arp_dataset, quoting=csv.QUOTE_NONNUMERIC)
            #wr.writerows(np.transpose(totalAvgPwr))
            wr.writerows([[lst] for  lst in data])
    
    def load_data(self, fileName):
        data = pd.read_csv(fileName, header=None,sep='\n')
        data = data[data.columns[0:]].values
        return data.transpose();
    
    def generateFreqData(self,lambda1,lambda2,numberOfSamples):
        totalPwrLvl = arp.generateFreqEnergy(arp,lambda1,lambda2,numberOfSamples)
        powerData = pd.DataFrame(np.array(totalPwrLvl).reshape(-1,1),totalPwrLvl)
        powerData = powerData[powerData.columns[0:]].values
        powerLvlLambda = powerData.transpose();
        return powerLvlLambda;
    
    def calculateAccuracy(self, score, powerLvl,sampleSize):
        dLen = 100 #length of the energy detector
        N = sampleSize
        N_train = dLen*N//2-dLen+1; #training data length - 14901
        #Training label ground truth/target
        wLen = 5*dLen; 
        N_test = dLen*N//2; #test data length - 15000
        N = N_train+N_test;
        
        totalPwr = 10*np.log10(np.abs(np.real(powerLvl[0,N_train+wLen:N])))-30
        prediction = 10*np.log10(np.abs(score))-30
        rmse = mean_squared_error(totalPwr,prediction)
        return rmse;
    
svmp = SVMPredictor()
numberOfSamples = 500
lambda1 = 0.8;
lambda2 = 0.8;
totalPwrLvl = svmp.generateFreqData(lambda1,lambda2,numberOfSamples)    
prediction = svmp.predictor(numberOfSamples,totalPwrLvl) 
svmp.plotSVM(totalPwrLvl, prediction,numberOfSamples)   

#arp_v = arp.generateFreqEnergy(arp,lambda1,lambda2,numberOfSamples)
#svmp.generatereqByLambda(lambda1, lambda2,numberOfSamples)

#totalPwrLvl = arp.generateFreqEnergy(arp,lambda1,lambda2,numberOfSamples)
#saveARPData(totalPwrLvl)
#totalPwrLvl = svmp.load_data("svmDataSet.csv")
#prediction = svmp.predictor(numberOfSamples,totalPwrLvl)
#svmp.plotSVM(totalPwrLvl,prediction,numberOfSamples)

'''
s.load_data("svmDataSet.csv")

'''
#s.plotSVM(totalPwrLvl,prediction,numberOfSamples)
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

numberOfSamples = 300
lambda1 = 0.8;
lambda2 = 0.8;

class SVMPredictor:
    def calcAccuracy(self, lambda1, lambda2):
        return 0;
    
    def predictor(self,numberOfSamples, datas):
        
        N = numberOfSamples
        dLen = 100 #length of the energy detector
        N_train = dLen*N//2-dLen+1; #training data length - 14901
        #Training label ground truth/target
        wLen = 5*dLen; 
        N_test = dLen*N//2; #test data length - 15000
        N = N_train+N_test; #total data length - 29901
        #plt.plot( np.array([i for i in np.arange(np.size(datas))]),data)
        sample_len = 5
        
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
        
        a = N_test
        
        
        for b in np.arange(a-wLen):
            for t in  np.arange(wLen):
                testData.itemset((t,b),np.real(datas[0,N_train+t+b]));

        
        label_reshape = trainLbl.reshape((N_train-wLen))
        train_reshape = trainData.transpose()
        
        #model = LogisticRegressionWithLBFGS.train(parsePoint(train_reshape,label_reshape))
        
        clf = svm.LinearSVR(epsilon=10e-90, C=10e6, max_iter=5000, dual=True,random_state=0,loss='squared_epsilon_insensitive',tol=10e-2,verbose=10).fit(train_reshape,label_reshape)
        fSteps = dLen; #tracks number of future steps to predict

        predicted = np.zeros((fSteps, N_test-wLen));
        obsSample = np.zeros((0, N_test-wLen));
        
        nData = testData;
        for i in np.arange(0,sample_len):
            predicted[i] = clf.predict(nData.transpose());
            nData = np.concatenate((testData[1:wLen:,],predicted[i:i+1,:]));
        
        predSet = self.generateMultistep(dLen,N_test-wLen,predicted,obsSample)

        return predicted
    def generateMultistep(self,fSteps, end,predicted,obsSample):
        predSet = np.zeros((fSteps, end))
        setCnt = 0;
        for i in np.arange(0,end):
            predSet[setCnt,i-setCnt:i-setCnt+fSteps] = predicted[:,i].transpose()
            if (setCnt+1)==fSteps:
                obsSample[i-setCnt] = 1;
                setCnt = 0;
            else:
                setCnt = setCnt + 1;
        return predSet
    def plotAccuracy(self, lambda_1,lambda_2):
        return 0;
    
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
        
        plt.figure(figsize=(10,10))
        
        plt.plot(subplot_range,totalPwr_score,subplot_range,prediction_score)
        plt.title('one-step ahead prediction')
        plt.legend(['Input Signal','Prediction'])
        plt.xlabel('Samples')
        plt.ylabel('Magnitude (dBm)')
        plt.show()

    def generatereqByLambda(self,lambda1, lambda2,numberOfSamples):
        lambda_range=0.01
        lst = len(np.arange(lambda_range,lambda1,lambda_range)) # Get the length of the lambda range
         
        lambda1_range = np.zeros((lst,lst))
        r = 0;
        
        fig = plt.figure(figsize=(30,30))

        for i in np.arange(lambda_range,lambda1,lambda_range):
            s = 0;
            for j in np.arange(lambda_range,lambda2, lambda_range):
                
                totalPwrLvl = arp.generateFreqEnergy(arp,i,j,numberOfSamples)
                powerData = pd.DataFrame(np.array(totalPwrLvl).reshape(-1,1),totalPwrLvl)
                powerData = powerData[powerData.columns[0:]].values
                powerLvlLambda = powerData.transpose()
                
                prediction = svmp.predictor(numberOfSamples,powerLvlLambda)
                
                score = prediction[1,:]
                accuracy = svmp.calculateAccuracy(score, powerLvlLambda, numberOfSamples)
                print("Lambda 1",i,"lambda2",j," -: Error Rate: ",accuracy);
                lambda1_range.itemset((r,s),accuracy/100)
                colors = lambda1_range[r,s]
                
                plt.scatter(i,j,c=colors, s=(accuracy)*10,alpha=0.55)
        
                s = s + 1;
            r = r+1;
        
        plt.colorbar()
        fig.savefig("lambda0.8range0.01_300samples.png")
        return 0;
    
    def saveARPData(self, data):
        with open("svmDataSet.csv", 'w') as arp_dataset:
            wr = csv.writer(arp_dataset, quoting=csv.QUOTE_NONNUMERIC)
            #wr.writerows(np.transpose(totalAvgPwr))
            wr.writerows([[lst] for  lst in data])
    
    def load_data(self, fileName):
        data = pd.read_csv(fileName, header=None,sep='\n')
        data = data[data.columns[0:]].values
        return data.transpose()
    
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
        rmse = mean_squared_error(prediction,totalPwr)
        return rmse;
    
svmp = SVMPredictor()
print(datetime.now())

print(datetime.now())
totalPwrLvl = arp.generateFreqEnergy(arp,lambda1,lambda2,numberOfSamples)
svmp.saveARPData(totalPwrLvl)
totalPwrLvl = svmp.load_data("svmDataSet.csv")
prediction = svmp.predictor(numberOfSamples,totalPwrLvl)
score = prediction[1,:]
accuracy = svmp.calculateAccuracy(score, totalPwrLvl, numberOfSamples)
print("Error Rate: ",accuracy);
svmp.plotSVM(totalPwrLvl,prediction,numberOfSamples)

print(datetime.now())
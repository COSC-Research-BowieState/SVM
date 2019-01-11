#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 11:13:11 2018

@author: hkyeremateng-boateng
"""
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
import pandas as pd
from ARPSimulator import ARPSimulator as arp
import csv
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from sklearn import preprocessing
rng = np.random.RandomState(42)
class SVMPredictor:
    def calcAccuracy(self, lambda1, lambda2):
        return 0;

    #@autojit
    def predictor(self,numberOfSamples, datas):
        
        N = numberOfSamples
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
        '''
        model = self.svm_model()
        
        model.fit(train_reshape,label_reshape,epochs=10, batch_size=50,validation_split=0.5)

        y = model.predict(testData.transpose())
        print(y)
        print(y.transpose().shape)
        '''
        clf = self.fit(train_reshape,label_reshape)        
        fSteps = dLen; #tracks number of future steps to predict

        predicted = np.zeros((fSteps, N_test-wLen));
 
        nData = testData;
        
        for i in np.arange(0,sample_len):
            predicted[i] = clf.predict(nData.transpose())
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
    
    #@autojit
    def fit(self,train_reshape,label_reshape):
        clf = svm.LinearSVR(epsilon=10e-90, C=10e5, max_iter=1000000,verbose=6, dual=True,random_state=0,loss='squared_epsilon_insensitive',tol=10e-2).fit(train_reshape,label_reshape)
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
        accuracy = self.calculateAccuracy(score, totalPwrLvl, numberOfSamples)
        accuracy_str = "One-step ahead prediction - Accuracy: "+str(100-accuracy)
        fig = plt.figure(figsize=(30,20))
        
        plt.plot(subplot_range,totalPwr_score,subplot_range,prediction_score)
        plt.title(accuracy_str)
        plt.legend(['Input Signal','Prediction'])
        plt.xlabel('Samples')
        plt.ylabel('Magnitude (dBm)')
        
        fig.savefig("3kSamples_0.8Lambda_1Miterations_epsilon_12.png")
        plt.show()

    def svm_model(self):
        model = Sequential()
        model.add(Dense(14, input_dim=500, init='normal', activation='relu'))
        #model.add(Dense(7, init='normal', activation='relu'))
        model.add(Dense(1,  activation='softmax'))
        model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
        return model  
    def generatereqByLambda(self,lambda1, lambda2,numberOfSamples):
        lambda_range=0.01
        lambda1_length = len(np.arange(lambda_range,lambda1,lambda_range)) # Get the length of the lambda 1 range
        lambda2_length = len(np.arange(lambda_range,lambda2,lambda_range))
        color_range = np.zeros(lambda1_length*lambda2_length)

        x_range=np.zeros(lambda1_length*lambda2_length)
        y_range=np.zeros(lambda1_length*lambda2_length)
        #lambda1_range = np.zeros((lambda1_length,lambda2_length))
        fig = plt.figure(figsize=(20,10))
        plt.subplot(1,1,1)
        a = 0
        for i in np.arange(lambda_range,lambda1,lambda_range):

            for j in np.arange(lambda_range,lambda2, lambda_range):

                powerLvlLambda = svmp.generateFreqData(i,j,numberOfSamples)
                
                prediction = svmp.predictor(numberOfSamples,powerLvlLambda)
                
                score = prediction[1,:]
                accuracy = svmp.calculateAccuracy(score, powerLvlLambda, numberOfSamples)

                color_range.itemset(a,(accuracy/100))
                x_range.itemset(a,i)
                y_range.itemset(a,j)
                a =a+1
                
        plt.scatter(x_range,y_range,c=color_range, s=10*25, alpha=0.15,cmap='hsv')

        #plt.pcolormesh(x_range,y_range,color_range,cmap='hsv',norm=mcolors.Normalize(vmin=color_range.min(),vmax=color_range.max()))        
        cb = plt.colorbar()
        cb.set_label(" Prediction Accuracy % ")
        fig.savefig("0.8by0.8lambda700kitr3ksamples.png")
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
numberOfSamples =3000
lambda1 = 0.8;
lambda2 = 0.8;
#totalPwrLvl = svmp.generateFreqData(lambda1,lambda2,numberOfSamples)    
print("Prediction...",datetime.now())
##prediction = svmp.predictor(numberOfSamples,totalPwrLvl) 
#print("Prediction...ended ",datetime.now())
#svmp.plotSVM(totalPwrLvl, prediction,numberOfSamples)   

#arp_v = arp.generateFreqEnergy(arp,lambda1,lambda2,numberOfSamples)
svmp.generatereqByLambda(lambda1, lambda2,numberOfSamples)

'''
totalPwrLvl = arp.generateFreqEnergy(arp,lambda1,lambda2,numberOfSamples)
saveARPData(totalPwrLvl)
totalPwrLvl = s.load_data("svmDataSet.csv")
prediction = s.predictor(numberOfSamples,totalPwrLvl)
s.plotSVM(totalPwrLvl,prediction,numberOfSamples)
0242503182

s.load_data("svmDataSet.csv")

'''
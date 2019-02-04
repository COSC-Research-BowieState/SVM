# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 23:09:34 2018

@author: Hubert Kyeremateng-B
"""
import numpy as np
#from ARPSimulator import ARPSimulator as arp
import pandas as pd
from ARPSimulator import ARPSimulator as arp
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import tensorflow as tf
from tensorflow.python.framework import ops
from datetime import datetime
ops.get_default_graph()
tf.reset_default_graph()
session = tf.Session()

###############################################################################################
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

index = 0
for i in np.arange(wLen,N_train-1):
    trainLbl.itemset(index,powerLvlLambda[0,i]);
    index = index +1;
    

#Traing and test input data
trainData = np.zeros((wLen,N_train-wLen));
testData = np.zeros((wLen,N_test-wLen));
###### totalAvgPwr is one-dimensional array#####
###### trainData in a multi-dimensional array######

for s in np.arange(index):
    for k in  np.arange(wLen-1):
        trainData.itemset((k,s),np.real(powerLvlLambda[0,s+k]))

a = N_test


for b in np.arange(a-wLen):
    for t in  np.arange(wLen):
        testData.itemset((t,b),np.real(powerLvlLambda[0,N_train+t+b]));

label_reshape = trainLbl.reshape((N_train-wLen))
train_reshape = trainData.transpose()
################################################################################################
print("Starting tensorflow")
#Declare Tensorflow batch size
batch_size = 50
#Declaring Placeholderss
train_data = tf.placeholder(shape=[500,None], dtype=tf.float32, name="train")
trainLbl_tf = tf.placeholder(shape=[1,None], dtype=tf.float32, name="training_label")
test_data = tf.placeholder(shape=[None,1], dtype=tf.float32, name="test_data")

# Create variables for linear regression
trainLbl_variable = tf.Variable(trainLbl.astype(np.float32), name="trainLbl_variable")
trainData_variable = tf.Variable(trainData.astype(np.float32), name="trainData_variable")
testData_variable = tf.Variable(testData.astype(np.float32), name="testData_variable")

# Linear Kernel
my_kernel = tf.matmul(train_data, trainData_variable)

model_output = tf.add(my_kernel,trainLbl_variable)
print(my_kernel)
print(trainLbl_variable)
# Declare loss function

# = max(0, abs(target - predicted) + epsilon)
# 1/2 margin width parameter = epsilon
epsilon = tf.constant([0.5])
# Margin term in loss

loss = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(tf.subtract(model_output, trainData_variable)), epsilon)))
print("loss",loss)

init = tf.global_variables_initializer()
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)

train_step =optimizer.minimize(loss)
session.run(init)
feed = {train_data:trainData}
for i in range(10000):
    session.run(train_step,feed_dict=feed)
'''
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
    
    @cuda
    def __sigmoid(self,x):
        x = x.astype(np.float32)
        return 1 / (1 + np.exp(-x))
    
    def __loss_function(self,h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def __predict_probs(self,X):
        return self.__sigmoid(np.dot(X))
    
    def predict(self,X, threshold=0.5):
        return self.__predict_probs(X) >= threshold
'''


'''
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
'''

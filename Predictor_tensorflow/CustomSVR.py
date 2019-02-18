#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 17:29:10 2019

@author: hkyeremateng-boateng
"""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import pandas as pd
from datetime import datetime
from scipy.stats import linregress
from statistics import mean

ops.reset_default_graph()
# Global variables
print(datetime.now())
epochs = 100
numberOfSamples = 500
N = numberOfSamples
dLen = 100 #length of the energy detector
N_train = dLen*N//2-dLen+1; #training data length - 14901

wLen = 5*dLen;
N_test = dLen*N//2; #test data length - 15000
N = N_train+N_test; #total data length - 29901
sample_len = 5

def load_data(fileName):
    data = pd.read_csv(fileName, header=None,sep='\n')
    data = data[data.columns[0:]].values
    return data.transpose()

trainData = np.zeros((wLen,N_train-wLen));
testData = np.zeros((wLen,N_test-wLen));
trainLbl = np.zeros((1,N_train-wLen));
formattedData = load_data("svmDataSet.csv")

trainFeatures = 0
for i in np.arange(wLen,N_train-1):
    trainLbl.itemset(trainFeatures,formattedData[0,i]);
    trainFeatures = trainFeatures +1;
   
for s in np.arange(trainFeatures):
    for k in  np.arange(wLen-1):
        trainData.itemset((k,s),np.real(formattedData[0,s+k]))
for b in np.arange(N_test-wLen):
    for t in  np.arange(wLen):
        testData.itemset((t,b),np.real(formattedData[0,N_train+t+b]));
       
def input_train():
    for s in np.arange(trainFeatures):
        for k in  np.arange(wLen-1):
            trainData.itemset((k,s),np.real(formattedData[0,s+k]))
    return trainData
numOfFeatures = trainFeatures+1;

# Declare batch size
batch_size = 50

# Initialize placeholders
train_data = tf.placeholder(shape=[1, None], dtype=tf.float32,name="train_data")
trainLbl_data = tf.placeholder(shape=[1, None], dtype=tf.float32,name="train_label")
test_data = tf.placeholder(shape=[500, None], dtype=tf.float32,name="test_label")

# Create variables for linear regression
A = tf.Variable(tf.random_normal(shape=[1,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))
Q = tf.Variable(tf.random_normal(shape=[numOfFeatures,1]))

# Declare model operations
model_outputs = tf.add(tf.matmul(train_data, Q), b)
# Declare loss function
epsilon = tf.constant([0.01])

# Margin term in loss
loss_1 = tf.reduce_mean(tf.maximum(0., tf.subtract(tf.abs(tf.subtract(model_outputs, train_data)), epsilon)))
loss_2 = tf.reduce_mean(tf.square(model_outputs-train_data))
# Declare optimizer
optimizer = tf.train.GradientDescentOptimizer(0.075).minimize(loss_2)

# Initialize variables
init = tf.global_variables_initializer()

train_xi = trainData[0,:]
train_xi = train_xi.reshape((1,-1))
slope_list =[]
fSteps = dLen; #tracks number of future steps to predict
score = np.zeros((1,N_test));
predicted = np.zeros((fSteps, N_test));
prediction = np.zeros((fSteps, N_test-wLen));
predictions = np.zeros((N_test-wLen));
predicteds = np.zeros((fSteps, N_test));
with tf.Session() as sess:
    sess.run(init)

    # Training loop
    for i in range(epochs):
         train_x = trainData
         trainLbl_x = trainLbl
         test_x = testData
         sess.run(optimizer, feed_dict={train_data:train_xi,trainLbl_data:trainLbl_x})

    # Extract Coefficients
    slope = sess.run(Q)
    y_intercept = sess.run(b) #Recheck if the y intercept has to be calculated for each data point
    width = sess.run(epsilon)
    #Convert slope into a list (float)
    for i in np.arange(len(slope)):
        slope_list.append(slope[i][0].item())
    mean_xi = mean(slope_list)
    test_list =  testData[0,:].reshape((1,-1))
    
    for i in np.arange(24401):
        
        predictions[i] = (slope_list[i]*test_list[0][i])+y_intercept
predictions = predictions.reshape((1,-1))
score = predictions[0,:]
totalPwr = 10*np.log10(np.abs(np.real(formattedData[0,N_train+wLen:N])))-30
prediction = 10*np.log10(np.abs(score))-30
# Plot loss over time
plt.plot(totalPwr, 'k-', label='Train Data')
plt.plot(prediction, 'r-', label='Test Data')
plt.title('One-step ahead prediction')
plt.legend(['Input Signal','Prediction'])
plt.xlabel('Samples')
plt.ylabel('Magnitude (dBm)')
plt.show()
print(datetime.now())
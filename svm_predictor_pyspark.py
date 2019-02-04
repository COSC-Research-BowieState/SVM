#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 18 16:57:33 2018

@author: hkyeremateng-boateng
"""

from pyspark import SparkContext 
import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
import pandas as pd
from datetime import datetime
from ARPSimulator import ARPSimulator as arp
#from SVMPredictor import SVMPredictor as svmp
import csv
from sklearn.metrics import mean_squared_error
from pyspark.ml.regression import LinearRegression
from pyspark import SparkContext as sc

lambda1 = 0.8
lambda2 = 0.8
numberOfSamples = 300

lambda_range=0.1

sc.stop(sc)

data = arp.generateFreqEnergy(arp,lambda1,lambda2,numberOfSamples);
#prediction = svmp.predictor(numberOfSamples,dataj
sl = LinearRegression(fitIntercept=True,maxIter=100,solver="auto",epsilon=10e12,loss="squaredError",tol=1e-12,featuresCol = 'features')
data = data.reshape(1,-1);
rdd = sc.parallelize(data)
predict = sl.fit(rdd)
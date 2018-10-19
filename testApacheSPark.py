#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 20:56:06 2018

@author: hkyeremateng-boateng
"""
import numpy as np
from pyspark.mllib.classification import SVMWithSGD, SVMModel
from pyspark.mllib.regression import LabeledPoint
import pandas as pd

sample_len = 5
data = pd.read_csv('svmDataSet.csv', header=None,sep='\n')
data = data[data.columns[0:]].values
parsedData = data.transpose()


# Build the model
model = SVMWithSGD.train(parsedData, iterations=100)

# Evaluating the model on training data
labelsAndPreds = parsedData.map(lambda p: (p.label, model.predict(p.features)))
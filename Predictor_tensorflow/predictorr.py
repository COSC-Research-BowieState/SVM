#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 22:01:47 2018

@author: hkyeremateng-boateng
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn.svm as svm
import pandas as pd
from datetime import datetime
from ARPSimulator import ARPSimulator as arp
import tensorflow as tf
from sklearn.metrics import mean_squared_error

#tf.enable_eager_execution()
model = tf.keras.Sequential()


numberOfSamples = 3000
lambda1 = 0.8;
lambda2 = 0.8;

totalPwrLvl = arp.generateFreqEnergy(arp,lambda1,lambda2,numberOfSamples)

pwrLvlSlice = tf.data.Dataset.from_tensor_slices(totalPwrLvl)
pwrLvlItr = pwrLvlSlice.make_one_shot_iterator().get_next()

with tf.Session() as sess:
    



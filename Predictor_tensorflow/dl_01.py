#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 17:40:58 2018

@author: hkyeremateng-boateng
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:, (2,3)]
y = (iris.target == 0).astype(np.int)

per_clf = Perceptron(random_state=42,max_iter=10000)
per_clf.fit(X, y)

pred = per_clf.predict([[2,0.5]])
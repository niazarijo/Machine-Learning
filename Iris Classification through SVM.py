# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 13:25:42 2022

@author: niaz
"""

import torch
from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC  


iris = datasets.load_iris() #We load Iris Dataset
data = iris.data[:]             
target = np.array(iris.target)  #loading class label

n_samples = data.shape[0]
n_val = int(0.2 * n_samples) #Take 20% test data set

shuffled_indices = torch.randperm(n_samples) #suffling indices of dataset

train_indices = shuffled_indices[:-n_val]  
test_indices = shuffled_indices[-n_val:] #Take 20% test data 

train_data = data[train_indices] #Take 80% train data
train_label =target[train_indices]

test_data = data[test_indices]  #Take 20% test data
test_label= target[test_indices]


model = SVC(kernel='linear') #SVM model is created
model.fit(train_data,train_label)  #Model training 
pred = model.predict(test_data)  #Model Prediction

print('Accuracy of Model : {:.5f} %'.format(accuracy_score(pred,test_label)*100))


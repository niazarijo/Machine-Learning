# -*- coding: utf-8 -*-
"""
Created on Sat Jul 23 00:15:31 2022

@author: niaz
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

iris = datasets.load_iris() #We load Iris Dataset
data = iris.data[:, 2:]  # We take Petal Length and Petal Width

plt.scatter(data[:,0],data[:,1],color='red')

Xsum=np.sum(data[:,0])
Ysum=np.sum(data[:,1])

Xmu=np.mean(data[:,0])
Ymu=np.mean(data[:,1])

X=np.linspace(0,8,100)

#finding the values of the regression line coefficients by using Least Square Method 
#for fitting regression line
m = sum((data[:,0]-Xmu)*(data[:,1]-Ymu))/sum((data[:,0]-Xmu)**2)
b=Ymu-m*Xmu

plt.plot(X,m*X+b,color='black')
plt.show()
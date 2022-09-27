# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:40:00 2022

@author: niaz
"""

import numpy as np
import matplotlib.pyplot as plt

x1     = np.array([1,1,0,2,3,0,2,5,5,2,4,5,4,3])     # x1 component of each train feature vector
x2     = np.array([1,4,0,0,0,2,2,1,2,4,4,5,3,3])     # x2 component of each train feature vector

#x1 and x2 components as a matrix with additonal 1s for bias term
X = np.array([[1,1,1],[1,1,4],[1.,0,0],[1,2,0],[1,3,0],[1,0,2],[1,2,2],[1,5,1],[1,5,2],[1,2,4],[1,4,4],[1,5,5],[1,4,3],[1,3,3]])

y      = np.array([0,0,	0,	0,	0,	0,	0,	1,	1,	1,	1,	1,1,1])    # label of each train vector

Weights = np.array([1,0,0])

c_map  = ["green" if i == 1 else "red" for i in y]


plt.scatter(x1, x2,color=c_map) 


#logistic Regression Model

def sigmoid(z):
        
    y = 1./(1.+np.exp(-z))
    return y

#logistic Regression Loss Function
def lossFunction(X,h,y):
    
    
    m=len(y)
    cost= (1/m)*(-y.transpose().dot(np.log(h))-(1-y).transpose().dot(np.log(1-h)))
    
    grad=(1/m)*X.transpose().dot(h-y);

    return (cost,grad)
    

cost=1
learning_rate =1e-1

#logistic Regression Model Training
while (cost >0.05): 
   
    z=X.dot(Weights)
    a = sigmoid(z)
    
    cost,grad = lossFunction(X,a, y)    
    Weights=Weights-learning_rate*grad #updating parameters
    
    print('Training Loss: {:.5f} '.format(cost))  #Loss of the model 
    #print(Weights)
    
#Fitting Classification Line by finding slope and intercept by slope intercept line equation
x = np.arange(0,6,0.1)
m=-Weights[1]/Weights[2]
b=-Weights[0]/Weights[2]
y = m*x+b

plt.plot(x,y)

plt.show()


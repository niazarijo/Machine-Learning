# -*- coding: utf-8 -*-
"""
Created on Sat Sep 24 13:25:42 2022

@author: niaz

This model gives accuracy of 98% with 20 epochs only

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torchvision import datasets, transforms

mnist_train = datasets.MNIST(root='./', train=True, download=True, transform=transforms.ToTensor()) #downloading train data set and transforming it to a tensor for processing
mnist_test = datasets.MNIST(root='./', train=False, download=True, transform=transforms.ToTensor()) #downloading test data set and transforming it to a tensor for processing

train_loader = torch.utils.data.DataLoader(mnist_train,batch_size=10) #train data set of 60000 and batch size=10
test_loader = torch.utils.data.DataLoader(mnist_test,batch_size=10) #test data set of 10000 and batch size=10


class MnistModel(nn.Module):
    def __init__(self): #constructor
        super( MnistModel,self).__init__() #calling constructor of the parent class (nn.Module)
        
        hidden_1 = 400
        hidden_2 = 400
        hidden_3 = 400
        hidden_4 = 400
        
        self.hl1 = nn.Linear(28 * 28, hidden_1)  # hidden layer1 (784 -> hidden_1)
        self.hl2 = nn.Linear(hidden_1, hidden_2) # hidden layer2 (hidden_1 -> hidden_2)
        self.hl3 = nn.Linear(hidden_2, hidden_3) #hidden layer3 (hidden_2 -> hidden_3)
        self.hl4 = nn.Linear(hidden_3, hidden_4) #hidden layer4 (hidden_3 -> hidden_4)
        
        self.fl = nn.Linear(hidden_4, 10)        #final layer (hidden_4 -> 10)
                
        self.dropout = nn.Dropout(0.2) # dropout function with (p=0.2)


    def forward(self, x):
        
        a = x.view(-1, 28 * 28) # flattening image         
        a = F.relu(self.hl1(a)) # adding hidden layer1 with relu activation function        
        a = self.dropout(a)     # drop 20% of nodes     
        a = F.relu(self.hl2(a)) # adding hidden layer2 with relu activation function        
        a = self.dropout(a)     # drop 20% of nodes                
        a = F.relu(self.hl3(a)) # adding hidden layer3 with relu activation function        
        a = self.dropout(a)     # drop 20% of nodes 
        a = F.relu(self.hl4(a)) # adding hidden layer4 with relu activation function        
        a = self.dropout(a)     # drop 20% of nodes  
        a = self.fl(a)          # adding final layer
        return a


model = MnistModel() # initializing DNN
print(model)
model.train() #prepare for training

lossfunction = nn.CrossEntropyLoss() #loss function
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2) # stochastic gradient descent with learning rate = 0.01

n_epochs = 20

for epoch in range(n_epochs): # model training
    
    train_loss = 0.0
        
    for image, label in train_loader:
        
        optimizer.zero_grad() #set gradients to zeros        
        output = model(image) # computing predicted outputs        
        loss = lossfunction(output, label) # calculating the loss of model        
        loss.backward() # computing gradients of loss function        
        optimizer.step() # updating model parameters
        
        train_loss += loss.item() # train loss 
       
    print('Epoch: {} \tTraining Loss: {:.5f} on average'.format(epoch+1,train_loss/len(train_loader)))
    
test_loss = 0.0
prediction_list = []
predictions = []
model.eval() #prepare for testing

for data, target in test_loader:   # model testing
    
    output = model(data)    # computing predicted outputs    
    loss = lossfunction(output, target) # calculating the loss of model     
    test_loss += loss.item() # test loss     
    _, pred = torch.max(output, 1) # converting probabilities to predicted class
    
    
    correct = pred.eq(target).tolist() # comparing predicted values with target labels
    prediction_list.extend(correct) #Adding predicted values in a list
    
    predictions.append(accuracy_score(pred,target)) # comparing predicted values with target labels and find accuracy
   
    
# calculate average accuracy and test loss
print('Accuracy of Model : {:.5f} %'.format(sum(prediction_list)/len(prediction_list)*100)) #finding accuracy without using accuracy_score function

print('Accuracy of Model : {:.5f} %'.format(sum(predictions)/len(predictions)*100)) #finding accuracy by using accuracy_score function
print('Test Loss: {:.5f} on average'.format(test_loss/len(predictions)*100))

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:27:55 2023

@author: Rakshya
"""
#importing libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time

#working directory
path = 'C:/Users/14098/OneDrive - Lamar University/Desktop/Machine Learning/Project_4'
os.chdir(path)

#dataframe
df = pd.read_csv('concretedata.csv')

#Reclassifying the compressive strength by defining function
def safety_check(val):
    if val < 45:
        return 'Unsafe'
    elif val >= 45:
        return 'Safe'
    else:
        return None

# Replacing the compressive strength with reclassified one 
df['StrengthMPa'] = df['StrengthMPa'].apply(safety_check)


# Separating the input variables and outpot variable
X = df[['Cement', 'BlastFurnaceSlag', 'FlyAsh', 'Water', 'Superplasticizer',
       'CoarseAgg', 'FineAgg', 'AgeDays']].values
Y = df['StrengthMPa'].values


# Converting string data to binary data
Str_Binary = {'Unsafe': 0, 'Safe': 1}
Y = np.array([Str_Binary[label] for label in Y])

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=107)


# Define model architecture
class PavementConditionModel(nn.Module):
    def __init__(self):
        super().__init__() #initializing the layer through the neural network
        self.co1 = nn.Linear(8, 32) #input size of 8 that gives output size of 32 for first layer.
        self.co2 = nn.Linear(32, 16)#fully connected layer that reduces the tensor dimensionality to 16
        self.co3 = nn.Linear(16, 8)
        self.co4 = nn.Linear(8, 2)

    def forward(self, z):
        z = nn.functional.relu(self.co1(z))#Input x passed through the first layer with activation function Rectified Linear Unit (relu)
        z = nn.functional.relu(self.co2(z))
        z = nn.functional.relu(self.co3(z))
        z = nn.functional.softmax(self.co4(z), dim=1)#Output from 3rd layer passed through softmax for normalization of output
        return z

model = PavementConditionModel() 

##compute the negative log-likelihood loss between the predicted logits and the true labels and optimizer
loss_fun = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

#finding start time to check the total modeling time of the pytorch
starttime = time.time()

# Training model
for epoch in range(10):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(zip(X_train, Y_train)):
        inputs, labels = data
        inputs = torch.tensor(inputs).float().unsqueeze(0)
        labels = torch.tensor(labels).long().unsqueeze(0)
        # Backpropagation (minimize the error)
        optimizer.zero_grad()#removes the gradient of all optimized parameter

        outputs = model(inputs)
        loss = loss_fun(outputs, labels)
        loss.backward()# computes loss gradient


        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1) # Compute accuracy on training data
      
        total += 1
        correct += (predicted == labels).sum().item()
      
        train_accuracy = correct / total
        optimizer.step()#updates the parameter by subtracting the parameter 
        #with product of learning rate and gradient.

    print(f'Epoch {epoch + 1} loss: {running_loss / len(X_train)}', f'Test accuracy: {correct / total}')

#finding ending time and total time of model training    
endtime = time.time()
totaltime = endtime - starttime
print("Total time:", totaltime, "seconds")

# Evaluate model on test set
with torch.no_grad():# disabling gradient descent 
    correct = 0
    total = 0
    running_loss =0
    for i, data in enumerate(zip(X_test, Y_test)): #running training dataset row by row by looping
        inputs, labels = data
        inputs = torch.tensor(inputs).float().unsqueeze(0)#convert input to pytorch tensor, input tensor data type: float, additional dimension added by unsqueeze() function
        labels = torch.tensor(labels).long().unsqueeze(0)

        outputs = model(inputs)
        loss = loss_fun(outputs, labels) # Compute loss function on testing data
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test loss: {running_loss / len(X_test)},Test accuracy: {correct / total}')



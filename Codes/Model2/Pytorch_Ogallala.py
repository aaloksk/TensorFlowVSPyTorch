# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 21:10:35 2023

@author: Kushum
"""

# Step 1: Loading Libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time

#Step 2: Ste working directory
path = 'D:\\OneDrive - Lamar University\\00Spring2023\\MachineLearning\\Project_3\\WD'
os.chdir(path)

#Step 3: Loading the Ogallala datset 
df = pd.read_csv('Ogallala_Fluoride.csv')

#Step 4: Define a function to reclassify the values of fluoride 
def reclassify(val):
    if val <= 1.5:           # cuoff value for fluoride concentration is taken 1.5 mg/L
        return 'safe'
    else :
        return 'vulnerable'
    
    
#Step 5: Apply the function to create a new 'Vulnerability_State' column with the reclassified values
df['Vulnerability_State'] = df['Fluoride_avg'].apply(reclassify)
df.columns


#Step 6: Split data into input (x) and output (y) variables
X = df[['Latitude', 'WellDepth', 'Rainfall', 'Elevation', 'Tmin', 'claytotal_l', 'awc_l']].values
y = df['Vulnerability_State'].values


#Step 7: Convert reclassified class to integers values
Re_class = {'safe': 1, 'vulnerable': 0 }
y = np.array([Re_class[label] for label in y])


#Step 8: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=9841)


#Step 9: Define model architecture
class VulnerabilityModel(nn.Module):
    def __init__(self):
        super().__init__()             #Initializes the layers of the model architecture
        self.fc1 = nn.Linear(7, 32)   #Inputs tensor of size 11, output tensor of size 32
        self.fc2 = nn.Linear(32, 16)   #Second fully connected layer that reduces the dimensionality of the tensor to 16
        self.fc3 = nn.Linear(16, 8)    #Third fully connected layer, reduces dimensionality of the tensor furthur to 8
        self.fc4 = nn.Linear(8, 2)     # Final layer (fully connecyed layer), 2 is number of class used

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))        #Input x passed through the 'first fully connected' layer, activation function Rectified Linear Unit (relu)
        x = nn.functional.relu(self.fc2(x))        #Output from layer one is paased as input to 'second fully connected layer', activation function= relu
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.softmax(self.fc4(x), dim=1)   #Output from 3rd layer passed through softmax for normalization of output
        return x

model = VulnerabilityModel()


#Step 10: Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

#Starting time (to calculate time to train and test dataset)
start_time = time.time()

#Step 11: Train model
for epoch in range(8):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(zip(X_train, y_train)):          #to run training dataset row by row (loop of training)
        inputs, labels = data
        inputs = torch.tensor(inputs).float().unsqueeze(0)    #convert input to pytorch tensor, input tensor data type: float, additional dimension added by unsqueeze() function
        labels = torch.tensor(labels).long().unsqueeze(0)     #Convert lable to pytorch tensor, datatype: Long interger, additinal dimension

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)                      # Compute loss function on training data
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)            # Compute accuracy on training data
        
        total += 1
        correct += (predicted == labels).sum().item()
        
        train_accuracy = correct / total
    
    print(f'Epoch {epoch + 1} loss: {running_loss / len(X_train)}', f'Test accuracy: {correct / total}')   # to print loss function and accuracy at each epoch
    
end_time = time.time()

#Step 13: Time to run the model
total_time = end_time - start_time
print("Total time: ", total_time, " seconds.")


#Step 12: Evaluate model on test dataset 
with torch.no_grad():
    correct = 0
    total = 0
    running_loss = 0
    for i, data in enumerate(zip(X_test, y_test)):
        inputs, labels = data
        inputs = torch.tensor(inputs).float().unsqueeze(0)
        labels = torch.tensor(labels).long().unsqueeze(0)
        
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)                      # Compute loss function on testing data
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Test loss: {running_loss / len(X_test)}, Test accuracy: {correct / total}") 










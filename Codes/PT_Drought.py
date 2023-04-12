# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 23:49:06 2023

@author: reena
"""

# Loading libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time

#Setting working directory
path = 'C:\\Users\\reena\\OneDrive - Lamar University\\Desktop\\Machine learning\\Project4'
os.chdir(path)

#Reading the dataset
df = pd.read_csv('Drought.csv')

#Function to reclassify the drought values
def reclass_drought(val):
    if val >= 4:
        return 'Extreme Drought'   #Extreme drought if greater than or equal to 4
    elif val <4 and val >2:
        return 'Moderate Drought'  #Moderate drought if between 2 and 4
    elif val <=2:
        return 'No Drought'  # No drought if less than or equal to 2
    else:
        return None

 
# Apply function to the Drought level index column and create a new column with the reclassified values
df['Drought_Level_Index'] = df['Drought_Level_Index'].apply(reclass_drought)


# Split data into input (X) and output (y) variables
X = df[['Precipitation_in', 'Temperature_C',
       'Vegetation', 'Evapotranspiration']].values
Y = df['Drought_Level_Index'].values



# Convert class labels to integers
value_to_int = {'No Drought': 0, 'Moderate Drought': 1, 'Extreme Drought': 2}
Y = np.array([value_to_int[label] for label in Y])


# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2055)


# Defining the model architecture
class DroughtPredictModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.f1 = nn.Linear(4, 64)
        self.f2 = nn.Linear(64, 32)
        self.f3 = nn.Linear(32, 16)
        self.f4 = nn.Linear(16, 3)

    def forward(self, a):
        a = nn.functional.relu(self.f1(a))
        a = nn.functional.relu(self.f2(a))
        a = nn.functional.relu(self.f3(a))
        a = nn.functional.softmax(self.f4(a), dim=1)
        return a

model = DroughtPredictModel()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

#Get Start time of the train model
start_time = time.time()

# Training model
for epoch in range(7):
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(zip(X_train, Y_train)):
        inputs, labels = data
        inputs = torch.tensor(inputs).float().unsqueeze(0)
        labels = torch.tensor(labels).long().unsqueeze(0)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1) # Compute accuracy on training data
        
        total += 1
        correct += (predicted == labels).sum().item()
        
        train_accuracy = correct / total
    
    print(f'Epoch {epoch + 1} loss: {running_loss / len(X_train)}')
    print(f'Train accuracy: {correct / total}')

# Evaluate model on test set
with torch.no_grad():
    correct = 0
    total = 0
    for i, data in enumerate(zip(X_test, Y_test)):
        inputs, labels = data
        inputs = torch.tensor(inputs).float().unsqueeze(0)
        labels = torch.tensor(labels).long().unsqueeze(0)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += 1
        correct += (predicted == labels).sum().item()

    print(f"Test loss: {running_loss / len(X_test)}, Test accuracy: {correct / total}")

#Get end time to run the model    
end_time = time.time()

#Calculate the total time required to run the model
total_time = end_time - start_time
print("Total time: ", total_time, " seconds.")
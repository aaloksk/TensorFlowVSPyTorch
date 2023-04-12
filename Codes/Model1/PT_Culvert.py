# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 14:27:55 2023

@author: Aalok
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time

path = 'C:\\Users\\Aalok\\OneDrive - lamar.edu\\0000CVEN6301_ML\\Project4'
os.chdir(path)

df = pd.read_csv('TXculvertdata.csv')


# Define a function to reclassify the values
def reclassify(val):
    if val in [0, 1]:
        return 'Failed'
    elif val in [2, 3]:
        return 'Critical'
    elif val in [4, 5]:
        return 'Poor'
    elif val in [6, 7]:
        return 'Good'
    elif val in [8, 9]:
        return 'Excellent'
    else:
        return None   # handle any other value if exists

# Apply the function to the 'CULVERT_COND_062' column and create a new column with the reclassified values
df['CULVERT_Condition'] = df['CULVERT_COND_062'].apply(reclassify)

df.columns

df['SVCYR2'] = df['SVCYR']**2 # Add SVCYR square to the dataset

# Split data into input (X) and output (y) variables
X = df[['ADT', 'SVCYR', 'Reconst', 'PTRUCK', 'SVCYR2']].values
y = df['CULVERT_Condition'].values



# Convert class labels to integers
class_to_int = {'Failed': 0, 'Critical': 1, 'Poor': 2, 'Good': 3, 'Excellent': 4}
y = np.array([class_to_int[label] for label in y])


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define model architecture
class BridgeConditionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 5)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.softmax(self.fc4(x), dim=1)
        return x

model = BridgeConditionModel()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Train model
for epoch in range(10):
    running_loss = 0.0
    for i, data in enumerate(zip(X_train, y_train)):
        inputs, labels = data
        inputs = torch.tensor(inputs).float().unsqueeze(0)
        labels = torch.tensor(labels).long().unsqueeze(0)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()

    print(f'Epoch {epoch + 1} loss: {running_loss / len(X_train)}')

# Evaluate model on test set
with torch.no_grad():
    correct = 0
    total = 0
    for i, data in enumerate(zip(X_test, y_test)):
        inputs, labels = data
        inputs = torch.tensor(inputs).float().unsqueeze(0)
        labels = torch.tensor(labels).long().unsqueeze(0)

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        total += 1
        correct += (predicted == labels).sum().item()

    print(f'Test accuracy: {correct / total}')
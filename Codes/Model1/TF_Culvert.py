# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 03:29:40 2023

@author: Aalok
"""



import os
import numpy as np
import pandas as pd
import tensorflow as tf
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
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(5,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')
])


#Defining hyperparameters
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.001
batch_size = 32

# Compile model with SGD optimizer #stochastic gradient descent (SGD)
optimizer = tf.keras.optimizers.SGD(
    learning_rate=learning_rate,
    momentum=momentum,
    decay=weight_decay
)

#Directly giving sgd
#model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Compining the model with optimizer and loss function
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Defining number of epochs
epochs = 5


start_time = time.time()

# Train model
model.fit(X_train, y_train, epochs=10, validation_split=0)

end_time = time.time()

total_time = end_time - start_time
print("Total time: ", total_time, " seconds.")

# Evaluate model on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)

























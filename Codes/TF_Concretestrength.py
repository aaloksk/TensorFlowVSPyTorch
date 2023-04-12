# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 03:29:40 2023

@author: Rakshya
"""
#Importing libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import time

#Setting working directory
path = 'C:/Users/14098/OneDrive - Lamar University/Desktop/Machine Learning/Project_4'
os.chdir(path)

#Reading csv and converting it to dataframe
df = pd.read_csv('concretedata.csv')

#Reclassifying the compressive strength by defining function
def safety_check(val):
    if val < 45: #the values above 45MPa is considered safe
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

# get the number of rows in the DataFrame
num_rows = df.shape[0] #checking the number of rows of dataframe

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=107)

# Define model architecture
#Model Architecure with sequential layer stacking
#One tensor input and one output and two linear layers with relu activation function
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(8,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])


# Compile model with optimizer and loss function
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

starttime = time.time()

# Fitting the model to adjust the model parameters and minimize the loss.
model.fit(X_train, Y_train, epochs=10, validation_split=0.2)

#finding ending time and total time of model training   
endtime = time.time()
totaltime = endtime - starttime
print("Total time:", totaltime, "seconds")

# Evaluate model on train set
train_loss, train_acc = model.evaluate(X_train, Y_train)
print('Train accuracy:', train_acc)

# Evaluate model on test set
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)




























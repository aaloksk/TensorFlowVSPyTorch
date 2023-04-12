# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 00:06:49 2023

@author: Kushum
"""

# Step 1: Loading Libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
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
model = tf.keras.models.Sequential([                                   # sequential model with 4 fully connected layers
    tf.keras.layers.Dense(32, activation='relu', input_shape=(7,)),   #Input size 32, activation function rectified linear unit, input shape: no of columns
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])


#Step 10: Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Step 11: Starting time (to calculate time to train and test dataset)
start_time = time.time()

#Step 12: Train model the model
model.fit(X_train, y_train, epochs=8, validation_split=0)

end_time = time.time()

#Step 11.1: Time to run the model
total_time = end_time - start_time
print("Total time: ", total_time, " seconds.")


#Step 13: Evaluate model on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test Loss:', test_loss, 'Test accuracy:', test_acc)






















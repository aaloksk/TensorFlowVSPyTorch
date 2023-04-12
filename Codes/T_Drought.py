# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 23:50:06 2023

@author: reena
"""

# Loading libraries
import os
import numpy as np
import pandas as pd
import tensorflow as tf
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
X = df[[ 'Precipitation_in', 'Temperature_C',
       'Vegetation', 'Evapotranspiration']].values
Y = df['Drought_Level_Index'].values



# Convert class labels to integers
value_to_int = {'No Drought': 0, 'Moderate Drought': 1, 'Extreme Drought': 2}
Y = np.array([value_to_int[label] for label in Y])


# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2055)

# Defining the model architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


# Compiling the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Get Start time of the train models
start_time = time.time()

# Train the model
model.fit(X_train, Y_train, epochs=7, validation_split=0.2)

train_loss, train_acc = model.evaluate(X_train, Y_train)
print('Train accuracy:', train_acc)

# Evaluate the model on test set
test_loss, test_acc = model.evaluate(X_test, Y_test)
print('Test accuracy:', test_acc)

#Get end time to run the model
end_time = time.time()

#Calculate the total time required to run the model
total_time = end_time - start_time
print("Total time: ", total_time, " seconds.")
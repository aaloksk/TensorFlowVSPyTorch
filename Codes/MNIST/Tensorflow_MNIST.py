# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 14:16:53 2023

@author: Aalok
"""

# Import libraries
import os
import numpy as np
import tensorflow as tf
import time

#Checking tensorflow version
print("TensorFlow version:", tf.__version__)

#Loading MNIST dataset
mnist = tf.keras.datasets.mnist


#Getting tht dataset as train and test
(x_train, y_train), (x_test, y_test) = mnist.load_data()


# Convert the sample data from integer to float
x_train, x_test = x_train / 255.0, x_test / 255.0         # Normlising 0 to 225 pixel value

#Model Architecure with sequentiallayer stacking
#One tensor input and one output and two linear layers with relu activation function
#Input image size for MNIST dta is 28pixel X 28pixels.
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)), #INput Layer of 28*28 images
  tf.keras.layers.Dense(512, activation='relu'), #512 neurons with ReLU activation
  tf.keras.layers.Dense(512, activation='relu'), #512 neurons with ReLU activation
  tf.keras.layers.Dense(10) #Output Layer
])
  

# Define the optimizer, loss function, and metrics
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #Loss Function
optimizer = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, decay=0.001) #SGD
metrics=['accuracy']

# Configuring and compiling the model using Keras (Before training)
model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)



start_time = time.time()

# Fitting the model to adjust the model parameters.
model.fit(x_train, y_train, epochs=10)

end_time = time.time()

total_time = end_time - start_time
print("Total time: ", total_time, " seconds.")


# Model evaluation using test sets to check model performance
model.evaluate(x_test,  y_test, verbose=2)
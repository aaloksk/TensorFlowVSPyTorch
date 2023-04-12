# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 14:11:33 2023

@author: Aalok
"""

import torch
print(torch.__version__)

from torch import nn
import torchvision


import torchvision.datasets as datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import torch.optim as optim
import time

training_data = datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor(),)
len(training_data)

test_data = datasets.MNIST(root='./data', train=False, download=True, transform=ToTensor(),)
len(test_data)


# Passinbg dataset as an argument to DataLoader (wraps an iterable over dataset, supports atomatic batching, sampling, suffling and multiprocessing data loading)
batch_size = 64    #batch size defined 64, each element in the dataloader iterable will return batch of 64 feature and labels


# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)


#Crating a model (Neural Network) from nn.model
# Define layer of network in "_ _ init_ _", "forward" function to specify how data will pass through the network and move to GPU of available to accelerate the operations
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# Optimiza the model parameters, use loss funtion and an optimizer to train dataset
loss_fn = nn.CrossEntropyLoss()


# Define hyperparameters
learning_rate = 0.001
momentum = 0.9
weight_decay = 0.001
batch_size = 32

# Create optimizer object
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, 
                      nesterov=False, weight_decay=weight_decay)



# Defining function to loop the model to make prediction using training dataset
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Keep track of training loss and accuracy
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_loss /= len(dataloader)
    accuracy = 100 * correct / size
    print(f"Training Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {train_loss:>8f} \n")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    accuracy = 100 * correct / size
    print(f"Test Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {test_loss:>8f} \n")



start_time = time.time()

#Training and testing the model
# printing model accuracy and loss at each epoch
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


end_time = time.time()

total_time = end_time - start_time
print("Total time: ", total_time, " seconds.")







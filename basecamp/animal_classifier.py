import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
# Use TensorDataset for preparing data for PyTorch models
# Allows us to store our X (features) and y (taret labels)
# as tensors, making them easy to manage
from torch.utils.data import TensorDataset
# DataLoader is a class that provides an iterable over the dataset
# It allows us to efficiently manage data loading during training
from torch.utils.data import DataLoader

# For the Dataloader we define two key parameters:
# batch_size: nhow many samples are included in each iteration
# this helps us process multiple samples at once
#
# shuffle: whether to shuffle the data at every epoch
# this randomizes the data order at each epoch, helping improve
# model generalization. One epoch is a full pass through
# the training dataloader and generalization means the model
# performs well on unseen data

# an epoch is one full pass through the training dataloader
# typical batch sizes are 32 or more for computational efficiency
batch_size = 2
shuffle = True

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Read the CSV file
animals = pd.read_csv(script_dir / "animal_dataset.csv")

# Define input features
# we use iloc to select all columns except the first (animal name)
# and the last (type which is our target variable)
features = animals.iloc[:, 1:-1]
# target is the last column
target = animals.iloc[:, -1]

# convert into a numpy array
# X is input features and y is target value
X = features.to_numpy()
y = target.to_numpy()

# input features X
print(X)
# Class labels for each animal: y
print(y)

# Instantiate dataset class
dataset = TensorDataset(torch.tensor(X), torch.tensor(y))

# then to access an individual sample, we use
# square bracket indexing like this
input_sample, label_sample = dataset[0]
print(input_sample)
print(label_sample)

# Create DataLoader instance with batch size and shuffling parameters
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# each element in the dataloader is a tuple of (inputs, labels)
# each iteration it selects two samples and their labels
for batch_inputs, batch_labels in dataloader:
    print('batch_inputs: ', batch_inputs)
    print('batch_labels:', batch_labels)
# Data Science Salaries dataset and test project
# Features: categorical, target: salary (USD)
# Problem type: Regression (continuous values)
# Final output layer: Linear layer
# Loss function: Mean Squared Error (MSE) (Regression specific)
# MSE is the mean of the squared difference between
# between predictions and ground truth values

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
# Use TensorDataset for preparing data for PyTorch models
# Allows us to store our X (features) and y (target labels)
# as tensors, making them easy to manage
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

# Get the directory where this script is located
script_dir = Path(__file__).parent

# Read the CSV file
salaries = pd.read_csv(script_dir / "datasci_salaries.csv")

# Define input features
# we use iloc to select all columns except the last (salary_in_usd which is our target)
features = salaries.iloc[:, :-1]
# target is the last column (salary_in_usd)
target = salaries.iloc[:, -1]

# convert into a numpy array
# X is input features and y is target value
X = features.to_numpy()
y = target.to_numpy()

# input features X
print("Features X:")
print(X)
# Target salary values: y
print("Target y:")
print(y)

# Instantiate dataset class
dataset = TensorDataset(
    torch.tensor(X).float(),
    torch.tensor(y).float()
)

# then to access an individual sample, we use
# square bracket indexing like this
input_sample, label_sample = dataset[0]
print("Sample input:", input_sample)
print("Sample label:", label_sample)

dataset = TensorDataset(
    torch.tensor(X).float(),
    torch.tensor(y).float()
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

model = nn.Sequential(
    nn.Linear(4, 2),
    nn.Linear(2, 1)
)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

num_epochs = 100
for epoch in range(num_epochs):
    for data in dataloader:
        # before forward pass, set gradients to zero
        optimizer.zero_grad()
        # get feature and target from data loader
        feature, target = data
        # forward pass
        prediction = model(feature).squeeze(-1)  # squeeze only last dim, keep batch dim
        # calculate loss
        loss = criterion(prediction, target)
        # backpropagate
        loss.backward()
        # update weights
        optimizer.step()
    
    # print loss for each epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    
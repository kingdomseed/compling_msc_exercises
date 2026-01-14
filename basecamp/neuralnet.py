import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# create input_tensor with three features
input_tensor = torch.tensor([[0.3471, 0.4547, -0.2356, 0.1234]])

# define our linear layer
linear_layer = nn.Linear(in_features=4, out_features=2)

# nn.Sequential container example
sequential_container = nn.Sequential(
    nn.Linear(in_features=4, out_features=3),
    nn.Linear(in_features=3, out_features=2),
    nn.Linear(in_features=2, out_features=1)
)

# pass the input_tensor through the linear layer
# to generate the output_tensor
# Our Linear layer receives our input_tensor as the parameter/argument
output_tensor = linear_layer(input_tensor)

print(output_tensor)

input_tensor = torch.tensor([[0.3471, 0.4547, -0.2356, 0.1234]])

container = nn.Sequential(
    nn.Linear(in_features=input_tensor.shape[-1], out_features=1),
    nn.Sigmoid()
)

output_tensor = container(input_tensor)
print(output_tensor)

# This shows the one hot encoding function in use
# F(y, y_hat) where tensor(0) is y
print(F.one_hot(torch.tensor(0), num_classes = 3))

sample = torch.tensor([0.3471, 0.4547, -0.2356, 0.1234, 0.2345, 0.1234, 0.3456, 0.2345, 0.1234, 0.3456, 0.2345, 0.1234, 0.3456, 0.2345, 0.1234, 0.3456])
target = torch.tensor([0, 1])
model = nn.Sequential(
    nn.Linear(in_features=16, out_features=8),
    nn.Linear(in_features=8, out_features=4),
    nn.Linear(in_features=4, out_features=2)
)

prediction = model(sample)

# Calculate the loss and gradients
criterion = nn.CrossEntropyLoss()
loss = criterion(prediction.double(), target.double())
loss.backward()

print(loss)
print(model[0].weight)
print(model[0].weight.grad)

optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer.step()

print(optimizer)

# Layer Initialization
layer = nn.Linear(64, 128)
print(layer.weight.min())
print(layer.weight.max())
print(layer.bias.min())
print(layer.bias.max())
import torch

# Given the following network, compute the cross entropy loss
# Layer 1 uses ReLU
# Layer 2 uses Softmax
# Correct Order
# Compute z^1
# Compute a^1
# Compute z^2
# Compute y

x = torch.tensor([1, 2], dtype=torch.float)
W1 = torch.tensor([
    [1, 1],
    [-1, 1],
    [1, 2]
    ],
    dtype=torch.float)
W2 = torch.tensor([
                    [0, 1, 0],
                   [1, -1, 0],
                   [0, 0, 1]
                    ],
    dtype=torch.float)
b1 = torch.tensor([0, 0, 0], dtype=torch.float)
b2 = torch.tensor([0, 0, 0], dtype=torch.float)

# Ground Truth
y = torch.tensor([0, 1, 0], dtype=torch.float)


def cross_entropy_loss(predictions, targets):
    return -torch.sum(targets * torch.log(predictions))

# --- Computation ---
# Compute z^1
z1 = torch.matmul(W1, x) + b1

# Compute a^1
a1 = torch.relu(z1)

# Compute z^2
z2 = torch.matmul(W2, a1) + b2

# Compute y (prediction)
y_pred = torch.softmax(z2, dim=0)

# Compute Loss
loss = cross_entropy_loss(y_pred, y)

print(f"Loss: {loss.item()}")






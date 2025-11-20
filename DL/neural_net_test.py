import torch
import torch.nn.functional as F

# Input x
x = torch.tensor([[1.], [0.]])

# Layer 1 weights and bias
W1 = torch.tensor([[1., 1.],
                   [0., 1.],
                   [-1., 0.]])
b1 = torch.tensor([[1.], [0.], [-1.]])

# Layer 2 weights and bias
W2 = torch.tensor([[1., 1., -1.],
                   [0., 1., 0.]])
b2 = torch.tensor([[0.], [1.]])

# --- Computation ---

# Layer 1: z1 = W1 * x + b1
z1 = torch.matmul(W1, x) + b1
# Activation 1: ReLU
a1 = F.relu(z1)

# Layer 2: z2 = W2 * a1 + b2
z2 = torch.matmul(W2, a1) + b2
# Activation 2: Softmax
y = F.softmax(z2, dim=0)

print("Layer 1 Output (a1):")
print(a1)
print("\nFinal Output (y):")
print(y)


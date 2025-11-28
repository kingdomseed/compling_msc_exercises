import torch
import torch.nn as nn

class MyModel(nn.Module):
    # Wx + b
    """
    Initializes a MyModel instance.

    Parameters
    ----------
    in_dim : int
        The input dimension of the layer. Default is 300.
    out_dim : int
        The output dimension of the layer. Default is 300.

    Returns
    -------
    None
    """
    def __init__(self, in_dim=300, out_dim=300):

        super(MyModel, self).__init__()
        self.wxb = nn.Linear(in_features=in_dim,out_features=out_dim)
        return

    """
    Forward pass through the model.

    Parameters
    ----------
    input_forward : torch.Tensor
        The input tensor to be forwarded.

    Returns
    -------
    torch.Tensor
        The output tensor of the forward pass.
    """
    def forward(self, input_forward):

        return self.wxb(input_forward)

audio = torch.randn((1000, 1000))


mdl =MyModel(1000, 1000)
print(mdl(audio))

for name, param in mdl.named_parameters():
    print(name, param.shape)


class MultiMyModel(nn.Module):
    def __init__(self, in_dim=300, hidden_dim=200, out_dim=2):
        super(MultiMyModel, self).__init__()
        self.nn1 = MyModel(in_dim, hidden_dim)
        self.ac1 = nn.ReLU()
        self.nn2 = MyModel(hidden_dim, out_dim)
        self.ac2 = nn.Sigmoid()
        self.nn3 = MyModel(out_dim, in_dim)

        return

    def forward(self, input_forward):
        nn1out = self.nn1(input_forward)
        ann1out = self.ac1(nn1out)
        nn2out = self.nn2(ann1out)
        ann3out = self.ac2(nn2out)
        nn3out = self.nn3(ann3out)
        return nn3out

audiomm = torch.randn((4, 300))


mdlmm = MultiMyModel(300, 200, 3)
print(mdlmm(audiomm))

for name, param in mdlmm.named_parameters():
    print(name, param.shape)


loss_fn = nn.BCEWithLogitsLoss()
ce_loss_fn = nn.CrossEntropyLoss()

# Generate output from model
out = mdlmm(audiomm)

# For CrossEntropyLoss, labels should be class indices (shape: [batch_size])
label = torch.tensor([1, 1, 0, 1], dtype=torch.long)  # Class indices for each sample
print(label.shape)
loss = ce_loss_fn(out, label)

# back prop
loss.backward()
# print for my multi layers
for name, param in mdlmm.named_parameters():
    print(name, param.shape)
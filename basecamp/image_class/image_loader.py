import torch
import matplotlib.pyplot as plt

from pathlib import Path
from torch import nn

from torchvision. datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

from torchmetrics import Recall, Precision

# We must convert images to tensors and apply
# a transformation to normalize them.
train_transforms = transforms.Compose([
    # add random hoirizontal flip for data augmentation
    transforms.RandomHorizontalFlip(),

    # add random rotation for data augmentation
    transforms.RandomRotation(degrees=45),

    transforms.RandomAutocontrast(),

    # ensure all images are the same size
    transforms.ToTensor(),
    transforms.Resize((128, 128)),
])

# Create a dataset using ImageFolder
script_dir = Path(__file__).resolve().parent
dataset_train = ImageFolder(
    str(script_dir / 'clouds_train'),
    transform=train_transforms
)

dataloader_train = DataLoader(
    dataset_train,
    shuffle=True,
    batch_size=1,
)

dataset_test = ImageFolder(
    str(script_dir / 'clouds_test'),
)

dataloader_test = DataLoader(
    dataset_test,
    shuffle=False,
    batch_size=1,
)

image, label = next(iter(dataloader_train))

# ([1, 3, 128, 128]) batch_size, channels, height, width
print(f'Image shape: {image.shape}, Label: {label}')

# Rearrange the image tensor for display
# squeeze removes the batch dimension
# permute changes the order of dimensions
image = image.squeeze().permute(1, 2, 0)

plt.imshow(image)
plt.show()


class Net(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, padding=1
            ),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1
            ),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        self.classifier = nn.Linear(
            in_features=64*32*32,
            out_features=num_classes
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x


net = Net(num_classes=7)

net.train()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for epoch in range(10):
    for images, labels in dataloader_train:
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        print(f'Epoch [{epoch+1}/10], '
              f'Loss: {loss.item():.4f}')

metric_precision = Precision(
    task="multiclass", num_classes=7, average="macro"
)
metric_recall = Recall(
    task="multiclass", num_classes=7, average="macro"
)
recall_per_class = Recall(task="multiclass", num_classes=7, average=None)
recall_micro = Recall(task="multiclass", num_classes=7, average="micro")
recall_macro = Recall(task="multiclass", num_classes=7, average="macro")
recall_weighted = Recall(task="multiclass", num_classes=7, average="weighted")

net.eval()
with torch.no_grad():
    for images, labels in dataloader_test:
        outputs = net(images)
        _, preds = torch.max(outputs, 1)
        metric_precision(preds, labels)
        metric_recall(preds, labels)

precision = metric_precision.compute()
recall = metric_recall.compute()

print(f'Precision: {precision:.4f}, '
      f'Recall: {recall:.4f}')
print(f'dataset_test.class_to_idx: {dataset_test.class_to_idx}')

# Dictioanry comprehension to get per class recall
# k: recall[v].itemO
#     for k, v
#     in dataset_test.class_to_idx.items()
# } # get per class recall

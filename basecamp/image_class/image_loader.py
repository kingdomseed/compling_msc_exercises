from pathlib import Path

from torchvision. datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

# We must convert images to tensors and apply
# a transformation to normalize them.
train_transforms = transforms.Compose([
    transforms.ToTensor(),

    # ensure all images are the same size
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

image, label = next(iter(dataloader_train))
print(f'Image shape: {image.shape}, Label: {label}')

from water_dataset import WaterDataset
from torch.utils.data import DataLoader
from network import WaterNetwork
import torch.nn as nn
import torch. optim as optim

dataset_train = WaterDataset(csv_path='/Users/jholt/development/CL_Python/basecamp/intermediate/water_potability.csv')

dataloader_train = DataLoader(dataset_train, batch_size=2, shuffle=True)

features, labels = next(iter(dataloader_train))
print(f"Features: {features}, InLabels: {labels}")

water_network = WaterNetwork()



criterion = nn. BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.01)

for epoch in range(1000):
    for features, labels in dataloader_train:
        optimizer.zero_gradO)
        outputs = net (features)
        loss = criterion(
            outputs, Labels. view(-1, 1)
        )
        loss.backward()
        optimizer.step()
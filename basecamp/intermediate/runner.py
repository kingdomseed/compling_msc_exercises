from water_dataset import WaterDataset
from torch.utils.data import DataLoader
from network import WaterNetwork

dataset_train = WaterDataset(csv_path='/Users/jholt/development/CL_Python/basecamp/intermediate/water_potability.csv')

dataloader_train = DataLoader(dataset_train, batch_size=2, shuffle=True)

features, labels = next(iter(dataloader_train))
print(f"Features: {features}, InLabels: {labels}")

water_network = WaterNetwork()
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class WaterDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__() # ensures this behaves like it's parent class
        df = pd.read_csv(csv_path)
        self.data = df.to_numpy()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        features = self.data[idx, :-1]
        label = self.data[idx, -1]
        return features, label
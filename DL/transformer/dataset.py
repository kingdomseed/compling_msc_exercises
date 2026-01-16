import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class StringReversalDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # use iloc to the the row
        return self.data.iloc[idx]['input'], self.data.iloc[idx]['output']

def get_dataloader(csv_file, batch_size=1, shuffle=False):
    dataset = StringReversalDataset(csv_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


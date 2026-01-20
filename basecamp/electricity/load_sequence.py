import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader


# Load the electricity consumption data
script_dir = Path(__file__).resolve().parent
data_path = script_dir / 'electricity_seq_data' / 'LD2011_2014.txt'

# Read the semicolon-delimited file
df = pd.read_csv(data_path, sep=';', decimal=',', parse_dates=[0], index_col=0)

# Drop any missing values
df = df.dropna()

print(f'Total data points: {len(df)}')
print(f'Date range: {df.index[0]} to {df.index[-1]}')
print(f'First few rows:\n{df.head()}')


# Train/test split (80/20) - temporal split for time series
train_size = int(len(df) * 0.8)
train_data = df[:train_size]
test_data = df[train_size:]

print(f'\nTrain size: {len(train_data)}, Test size: {len(test_data)}')


def create_sequence(df, seq_length):
    # x is for inputs
    # y is for targets
    xs, ys = [], []
    for i in range(len(df) - seq_length):
        # 1 is the second column of the dataframe
        # which contains the electricity consumption
        # values
        x = df.iloc[i:(i + seq_length), 1]
        y = df.iloc[i + seq_length, 1]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


# Use 24 time steps (6 hours of 15-min intervals) to predict next value
seq_length = 24

X_train, y_train = create_sequence(train_data, seq_length)
X_test, y_test = create_sequence(test_data, seq_length)

print(f'\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')


dataset_train = TensorDataset(
    torch.from_numpy(X_train).float(),
    torch.from_numpy(y_train).float()
)

dataset_test = TensorDataset(
    torch.from_numpy(X_test).float(),
    torch.from_numpy(y_test).float()
)

dataloader_train = DataLoader(
    dataset_train,
    shuffle=False,
    batch_size=32
)

dataloader_test = DataLoader(
    dataset_test,
    shuffle=False,
    batch_size=32
)

print(f'\nDataset ready!')
x_batch, y_batch = next(iter(dataloader_train))
print(f'X batch shape: {x_batch.shape}, y batch shape: {y_batch.shape}')

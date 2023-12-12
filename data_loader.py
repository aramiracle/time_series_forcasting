import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.preprocessing import StandardScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, data_dir, sequence_length):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.data_files = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith('.csv')]
        self.data, self.x_scalers, self.y_scaler = self._load_data()

    def _load_data(self):
        # Initialize empty lists for X and y scalers
        x_scalers = []
        y_scaler = StandardScaler()

        # Initialize an empty data array
        data = []
        for file_path in self.data_files:
            # Load data from CSV file and remove the 'date' column
            df = pd.read_csv(file_path).drop(columns='date')
            values = df.values

            # Scaling features using Standardization (mean and std normalization)
            x_scaler = StandardScaler()
            x_scalers.append(x_scaler)

            values[:, :-1] = x_scaler.fit_transform(values[:, :-1])
            values[:, -1] = y_scaler.fit_transform(values[:, -1].reshape(-1, 1)).flatten()

            # Create sequences with the specified length
            for i in range(len(values) - self.sequence_length + 1):
                sequence = values[i:i + self.sequence_length]
                data.append(sequence)

        return data, x_scalers, y_scaler

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Get a sample from the dataset
        sample = self.data[idx]
        x = torch.tensor(sample[:, :-1], dtype=torch.float32)
        y = torch.tensor(sample[-1, -1], dtype=torch.float32)
        return x, y

class MyDataLoader(DataLoader):
    def __init__(self, data_dir, sequence_length, batch_size):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        super(MyDataLoader, self).__init__(dataset=TimeSeriesDataset(data_dir, sequence_length), batch_size=self.batch_size)
        self.train_loader, self.validation_loader, self.test_loader, self.x_scalers, self.y_scaler = self._load_data()

    def _load_data(self):
        train_size = int(0.8 * len(self.dataset))
        validation_size = int(0.1 * len(self.dataset))
        # Split the dataset into training, validation, and test sets
        train_dataset = torch.utils.data.Subset(self.dataset, range(train_size))
        validation_dataset = torch.utils.data.Subset(self.dataset, range(train_size, train_size + validation_size))
        test_dataset = torch.utils.data.Subset(self.dataset, range(train_size + validation_size, len(self.dataset)))

        # Create data loaders for training, validation, and test sets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, validation_loader, test_loader, self.dataset.x_scalers, self.dataset.y_scaler

    def get_loaders(self):
        # Return the data loaders for training, validation, and test sets
        return self.train_loader, self.validation_loader, self.test_loader

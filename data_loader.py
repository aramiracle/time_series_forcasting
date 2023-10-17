import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesDataset(Dataset):
    def __init__(self, data_dir, sequence_length):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.data_files = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith('.csv')]
        self.data = self._load_data()

    def _load_data(self):
        # Initialize an empty data array
        data = []
        for file_path in self.data_files:
            # Load data from CSV file and remove the 'date' column
            df = pd.read_csv(file_path).drop(columns='date')
            data = df.values
            # Scaling features using Min-Max scaling
            scaler = MinMaxScaler()
            data[:-1] = scaler.fit_transform(data[:-1])
        return data

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)

    def __getitem__(self, idx):
        # Get a sample from the dataset
        sample = self.data[idx]
        x = np.array(sample[:-1], dtype=np.float32)
        y = np.array(sample[-1:], dtype=np.float32)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

class MyDataLoader(DataLoader):
    def __init__(self, data_dir, sequence_length, batch_size):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        super(MyDataLoader, self).__init__(dataset=TimeSeriesDataset(data_dir, sequence_length), batch_size=self.batch_size)
        self.train_loader, self.validation_loader, self.test_loader = self._load_data()

    def _load_data(self):
        train_size = int(0.6 * len(self.dataset))
        validation_size = int(0.2 * len(self.dataset))
        # Split the dataset into training, validation, and test sets
        train_dataset = torch.utils.data.Subset(self.dataset, range(train_size))
        validation_dataset = torch.utils.data.Subset(self.dataset, range(train_size, train_size + validation_size))
        test_dataset = torch.utils.data.Subset(self.dataset, range(train_size + validation_size, len(self.dataset)))

        # Create data loaders for training, validation, and test sets
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, validation_loader, test_loader

    def get_loaders(self):
        # Return the data loaders for training, validation, and test sets
        return self.train_loader, self.validation_loader, self.test_loader

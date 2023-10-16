import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Define your custom dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, data_dir, sequence_length):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.data_files = [os.path.join(data_dir, filename) for filename in os.listdir(data_dir) if filename.endswith('.csv')]
        self.data = self._load_data()

    def _load_data(self):
        for file_path in self.data_files:
            df = pd.read_csv(file_path).drop(columns='date')
            data = df.values
            # Scaling features
            scaler = MinMaxScaler()
            data[:-1] = scaler.fit_transform(data[:-1])
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        x = np.array(sample[:-1], dtype=np.float32)  # Convert to ndarray
        y = np.array(sample[-1:], dtype=np.float32)
        x = torch.tensor(x, dtype=torch.float32)  # Convert to tensor
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

# Define your hybrid LSTM-GRU model
class HybridRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, use_lstm=True, use_gru=True):
        super(HybridRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.use_lstm = use_lstm
        self.use_gru = use_gru

        if self.use_lstm:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        if self.use_gru:
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        if self.use_lstm and self.use_gru:
            lstm_out, _ = self.lstm(x)
            gru_out, _ = self.gru(x)
            out = lstm_out + gru_out
        elif self.use_lstm:
            lstm_out, _ = self.lstm(x)
            out = lstm_out
        elif self.use_gru:
            gru_out, _ = self.gru(x)
            out = gru_out
        out = self.fc(out)
        return out

# Define constants and hyperparameters
data_dir = 'data/main'
best_model_path = 'saved_models/best_model.pth'
saved_models_dir = 'saved_models'
sequence_length = 24 * 7
input_size = 6  # Number of features (HUFL, HULL, MUFL, MULL, LUFL, LULL)
hidden_size = 64
num_layers = 2
output_size = 1  # Assuming you want to forecast a single feature
batch_size = 100
learning_rate = 0.001
num_epochs = 300

# Create a custom dataset and data loader
dataset = TimeSeriesDataset(data_dir, sequence_length)
train_size = int(0.7 * len(dataset))
validation_size = int(0.15 * len(dataset))
train_dataset = torch.utils.data.Subset(dataset, range(train_size))
validation_dataset = torch.utils.data.Subset(dataset, range(train_size, train_size + validation_size))
test_dataset = torch.utils.data.Subset(dataset, range(train_size + validation_size, len(dataset)))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define a directory for saving models
os.makedirs(saved_models_dir, exist_ok=True)

saved_model_files = [f for f in os.listdir(saved_models_dir) if f.startswith("epoch_")]
if saved_model_files:
    # Extract and store all epoch numbers from the saved model file names
    epoch_numbers = [int(f.split('_')[1].split('.')[0]) for f in saved_model_files]

    # Find the maximum (latest) epoch number
    start_epoch = max(epoch_numbers)

    # Construct the path for the latest epoch model
    last_epoch_model_path = os.path.join(saved_models_dir, f"epoch_{start_epoch:03d}.pth")

    # Load the model from the last epoch
    model = HybridRNN(input_size, hidden_size, num_layers, output_size, use_lstm=True, use_gru=True)
    model.load_state_dict(torch.load(last_epoch_model_path))
    model.train()
    print(f"Loaded model from the last epoch: {last_epoch_model_path}")

else:
    start_epoch = 0
    model = HybridRNN(input_size, hidden_size, num_layers, output_size, use_lstm=True, use_gru=True)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

best_loss = float('inf')  # Initialize the best loss with a high value

# Training loop
for epoch in tqdm(range(start_epoch + 1, num_epochs + 1)):
    running_loss = 0
    for inputs, targets in train_loader:
        current_batch_size = inputs.size(0)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        running_loss += loss
        loss.backward()
        optimizer.step()
    running_loss /= current_batch_size
    print(f'Epoch [{epoch}/{num_epochs}] Training Loss: {running_loss:.4f}')
    
    # Calculate validation loss
    model.eval()
    validation_loss = 0
    with torch.no_grad():
        for val_inputs, val_targets in validation_loader:
            val_outputs = model(val_inputs)
            val_loss = criterion(val_outputs, val_targets)
            validation_loss += val_loss
    validation_loss /= len(validation_loader)
    print(f'Epoch [{epoch}/{num_epochs}] Validation Loss: {validation_loss:.4f}')
    model.train()  # Set the model back to training mode
    
    # Save the model if it has the best validation loss so far
    if validation_loss < best_loss:
        best_loss = validation_loss
        best_epoch = epoch
        best_model_path = os.path.join(saved_models_dir, 'best_model.pth')
        torch.save(model.state_dict(), best_model_path)
        print('$$$ Best model based on validation loss is saved. $$$')
    
    # Save the model's state_dict after each epoch
    epoch_model_path = os.path.join(saved_models_dir, f'epoch_{epoch:03d}.pth')
    torch.save(model.state_dict(), epoch_model_path)

# Print the best model epoch and validation loss
print(f'Best Model Epoch: {best_epoch}, Best Validation Loss: {best_loss:.4f}')

# Evaluation
best_model = HybridRNN(input_size, hidden_size, num_layers, output_size, use_lstm=True, use_gru=True)
best_model.load_state_dict(torch.load(best_model_path))
best_model.eval()
test_losses = []
with torch.no_grad():
    for test_input, test_target in test_loader:
        test_output = best_model(test_input)
        test_loss = criterion(test_output, test_target)
        test_losses.append(test_loss.item())

# Print the average test loss
avg_test_loss = np.mean(test_losses)
print(f'Average Test Loss: {avg_test_loss:.4f}')

# Visualization (for multiple examples)
best_model.eval()
num_batches_to_plot = 30  # Change this to the number of batches you want to plot

test_outputs = []
test_targets = []

for i, (test_input, test_target) in enumerate(test_loader):
    if i >= num_batches_to_plot:
        break

    test_output = best_model(test_input)
    test_outputs.append(test_output.detach().numpy())
    test_targets.append(test_target.numpy())

test_outputs = np.concatenate(test_outputs)
test_targets = np.concatenate(test_targets)

plt.plot(test_outputs, color='r', label="Hybrid Model Predictions")
plt.plot(test_targets, color='b', label="Actual")
plt.legend()
plt.show()

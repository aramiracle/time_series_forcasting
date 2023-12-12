# load_model.py

import os
import torch

from model import HybridRNN, Transformer

def load_model(saved_models_dir, input_size, hidden_size, num_layers, output_size, num_heads, model_type='HybridRNN'):
    # List all model files in the specified directory that start with "epoch_"
    saved_model_files = [f for f in os.listdir(saved_models_dir) if f.startswith("epoch_")]

    # Determine the model class based on the provided model_type
    if model_type == 'HybridRNN':
        model_class = HybridRNN
    elif model_type == 'Transformer':
        model_class = Transformer
    else:
        raise ValueError(f"Invalid model type: {model_type}")

    if saved_model_files:
        # Extract and store all epoch numbers from the saved model file names
        epoch_numbers = [int(f.split('_')[1].split('.')[0]) for f in saved_model_files]

        # Find the maximum (latest) epoch number
        start_epoch = max(epoch_numbers)

        # Construct the path for the latest epoch model
        last_epoch_model_path = os.path.join(saved_models_dir, f"epoch_{start_epoch:03d}.pth")

        # Instantiate the model based on the determined model_class
        model = model_class(input_size, hidden_size, output_size, num_layers, num_heads)

        # Load the model from the last epoch
        checkpoint = torch.load(last_epoch_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_loss = checkpoint['best_loss']
        model.train()
        print(f"Loaded {model_type} from the last epoch: {last_epoch_model_path}")

    else:
        # If no saved models are found, start from epoch 0 with an initial best loss set to infinity
        start_epoch = 0
        best_loss = float('inf')
        model = model_class(input_size, hidden_size, output_size, num_layers, num_heads)

    return start_epoch, best_loss, model

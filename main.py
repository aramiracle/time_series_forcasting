import os
import torch.nn as nn
from data_loader import MyDataLoader
from trainer import train_model
from evaluator import evaluate_model
from load_model import load_model

# Define the main function
def main():
    # Set the data directory and file paths
    data_dir = 'data/main'
    results_dir = 'results'
    model_type = "HybridRNN"  # Change this to "Transformer" if needed
    saved_models_dir = f'saved_models/{model_type}'
    best_model_path = f'{saved_models_dir}/best_model.pth'
    predictions_dir = 'prediction'

    # Define model hyperparameters
    sequence_length = 24 * 7  # One week of hourly data
    input_size = 6
    hidden_size = 64
    num_layers = 1
    output_size = 1
    batch_size = 256
    num_heads = 2
    learning_rate = 1e-3
    num_epochs = 50

    # Create the saved_models directory if it doesn't exist
    os.makedirs(saved_models_dir, exist_ok=True)

    # Initialize the data loader and get data loaders
    data_loader = MyDataLoader(data_dir, sequence_length, batch_size)
    train_loader, validation_loader, test_loader = data_loader.get_loaders()

    start_epoch, best_loss, model = load_model(saved_models_dir, input_size, hidden_size, num_layers, output_size, num_heads, model_type=model_type)
    criterion = nn.MSELoss()

    # Train the model
    train_model(model, criterion, train_loader, validation_loader, start_epoch, num_epochs, best_loss, best_model_path, learning_rate, saved_models_dir, predictions_dir)

    # Evaluate the model on the test data
    evaluate_model(model, test_loader, criterion, best_model_path, model_type, results_dir, num_batches_to_plot=4)

# Execute the main function when the script is run
if __name__ == "__main__":
    main()

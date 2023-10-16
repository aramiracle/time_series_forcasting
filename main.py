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
    best_model_path = 'saved_models/best_model.pth'
    saved_models_dir = 'saved_models'
    predicted_dir = 'prediction'

    # Define model hyperparameters
    sequence_length = 24 * 30  # 30 days of hourly data
    input_size = 6
    hidden_size = 256
    num_layers = 2
    output_size = 1
    batch_size = 1000
    learning_rate = 1e-3
    num_epochs = 300

    # Create the saved_models directory if it doesn't exist
    os.makedirs(saved_models_dir, exist_ok=True)

    # Initialize the data loader and get data loaders
    data_loader = MyDataLoader(data_dir, sequence_length, batch_size)
    train_loader, validation_loader, test_loader = data_loader.get_loaders()

    # Load a pre-trained model or initialize a new one
    start_epoch, best_loss, model = load_model(saved_models_dir, input_size, hidden_size, num_layers, output_size)
    criterion = nn.MSELoss()

    # Train the model
    train_model(model, criterion, train_loader, validation_loader, start_epoch, num_epochs, best_loss, best_model_path, learning_rate, saved_models_dir, predicted_dir)

    # Evaluate the model on the test data
    evaluate_model(model, test_loader, criterion, best_model_path, num_batches_to_plot=2)

# Execute the main function when the script is run
if __name__ == "__main__":
    main()

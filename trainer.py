import os
import shutil
import torch
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# Define a function to visualize time series predictions and save them
def visualize_time_series_predictions_and_save(target_series, predicted_series, predictions_dir, epoch, phase):
    os.makedirs(f'{predictions_dir}/{phase}', exist_ok=True)

    for i in range(len(target_series)):
        target_data = target_series[i]
        predicted_data = predicted_series[i]

        # Create a line plot for the time series data
        plt.figure(figsize=(10, 6))
        plt.plot(target_data, label='Target Series', marker='o', markersize=4)
        plt.plot(predicted_data, label='Predicted Series', marker='o', markersize=4)
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.title(phase)
        plt.legend()

        # Save the figure
        plt.savefig(os.path.join(f'{predictions_dir}/{phase}', f'prediction_epoch_{epoch:03d}_batch_{i}.png'))
        plt.clf()
        plt.close()

def train_model(model, criterion, train_loader, validation_loader, start_epoch, num_epochs, best_loss, best_model_path, learning_rate, saved_models_dir, predictions_dir):
    # Create the optimizer with the Adam optimizer and the specified learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    shutil.rmtree(predictions_dir, ignore_errors=True)

    train_losses = []  # List to store training losses
    validation_losses = []  # List to store validation losses

    # Training loop
    best_epoch = -1

    # Visualize train targets with model predictions and save
    train_target_series = []
    train_predicted_series = []

    for epoch in tqdm(range(start_epoch + 1, num_epochs + 1)):
        running_loss = 0
        for inputs, targets in train_loader:
            current_batch_size = inputs.size(0)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            train_target_series.append(targets.detach().numpy())
            train_predicted_series.append(outputs.detach().numpy())
            running_loss += loss
            loss.backward()
            optimizer.step()
        running_loss /= current_batch_size
        train_losses.append(running_loss.item())
        print(f'Epoch [{epoch}/{num_epochs}] Training Loss: {running_loss:.6f}')

        # Visualize time series predictions for the entire train set and save to the predictions directory
        visualize_time_series_predictions_and_save(train_target_series, train_predicted_series, predictions_dir, epoch, phase='train')

        # Calculate validation loss
        model.eval()
        validation_loss = 0
        target_series = []
        predicted_series = []

        with torch.no_grad():
            for val_inputs, val_targets in validation_loader:
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_targets)
                validation_loss += val_loss
                target_series.append(val_targets.cpu().numpy())
                predicted_series.append(val_outputs.cpu().numpy())

        validation_loss /= len(validation_loader)
        validation_losses.append(validation_loss.item())
        print(f'Epoch [{epoch}/{num_epochs}] Validation Loss: {validation_loss:.6f}')
        
        # Visualize time series predictions for the entire validation set and save to the predictions directory
        visualize_time_series_predictions_and_save(target_series, predicted_series, predictions_dir, epoch, phase='validation')
        
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
        torch.save({'model_state_dict': model.state_dict(), 'best_loss': best_loss}, epoch_model_path)

    # Print the best model epoch and validation loss
    print(f'Best Model Epoch: {best_epoch}, Best Validation Loss: {best_loss:.6f}')

    # Plot the training and validation losses
    plt.figure()
    plt.plot(range(start_epoch + 1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(start_epoch + 1, num_epochs + 1), validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')  # Save the loss plot as an image
    plt.show()

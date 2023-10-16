import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def train_model(model, criterion, train_loader, validation_loader, start_epoch, num_epochs, best_loss, best_model_path, learning_rate, saved_models_dir):
    # Create the optimizer with the Adam optimizer and the specified learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []  # List to store training losses
    validation_losses = []  # List to store validation losses

    # Training loop
    best_epoch = -1

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
        train_losses.append(running_loss.item())
        print(f'Epoch [{epoch}/{num_epochs}] Training Loss: {running_loss:.6f}')
        
        # Calculate validation loss
        model.eval()
        validation_loss = 0
        with torch.no_grad():
            for val_inputs, val_targets in validation_loader:
                val_outputs = model(val_inputs)
                val_loss = criterion(val_outputs, val_targets)
                validation_loss += val_loss
        validation_loss /= len(validation_loader)
        validation_losses.append(validation_loss.item())
        print(f'Epoch [{epoch}/{num_epochs}] Validation Loss: {validation_loss:.6f}')
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
        torch.save({'model_state_dict' : model.state_dict(),
                    'best_loss' : best_loss
                    }, epoch_model_path)

    # Print the best model epoch and validation loss
    print(f'Best Model Epoch: {best_epoch}, Best Validation Loss: {best_loss:.4f}')

    # Plot the training and validation losses
    plt.figure()
    plt.plot(range(start_epoch + 1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(start_epoch + 1, num_epochs + 1), validation_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')  # Save the loss plot as an image
    plt.show()

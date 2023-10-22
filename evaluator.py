import torch
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, criterion, best_model_path, num_batches_to_plot=1):
    # Load the best model weights
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    # Perform model evaluation
    test_losses = []
    with torch.no_grad():
        for test_input, test_target in test_loader:
            test_output = model(test_input)
            test_loss = criterion(test_output, test_target)
            test_losses.append(test_loss.item())

    # Calculate and print the average test loss
    avg_test_loss = np.mean(test_losses)
    print(f'Average Test Loss: {avg_test_loss:.4f}')

    # Visualization of model predictions
    model.eval()
    test_outputs = []
    test_targets = []

    for i, (test_input, test_target) in enumerate(test_loader):
        if i >= num_batches_to_plot:
            break

        test_output = model(test_input)
        test_outputs.append(test_output.detach().numpy())
        test_targets.append(test_target.numpy())

    test_outputs = np.concatenate(test_outputs)
    test_targets = np.concatenate(test_targets)

    # Plot and save the results
    plt.plot(test_outputs, color='r', label="Model Predictions")
    plt.plot(test_targets, color='b', label="Actual")
    plt.legend()
    plt.savefig('results.png')
    plt.close()

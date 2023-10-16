# HybridRNN Time Series Forecasting Project

This project implements a Hybrid Recurrent Neural Network (RNN) for time series forecasting. Time series forecasting is a common problem in various domains, including finance, weather prediction, and energy consumption analysis. This project aims to provide a robust solution for forecasting time series data using a hybrid model that combines both LSTM and GRU layers for improved performance.

## Project Structure

The project consists of several Python files, each with its specific purpose and functionality:

### `main.py`

The `main.py` file serves as the entry point of the project. It encapsulates the main workflow, which includes data setup, defining model hyperparameters, training the model, and evaluating its performance. Here's an overview of what this file does:

- Set the data directory and file paths, including the location of the training data and the path to save the best model.
- Define model hyperparameters such as sequence length, input size, hidden size, number of layers, output size, batch size, learning rate, and the number of training epochs.
- Create the `saved_models` directory to store model checkpoints.
- Initialize the data loader and obtain data loaders for training, validation, and testing.
- Load a pre-trained model if available or initialize a new model.
- Train the model using the specified criterion (mean squared error) and save the best model based on the validation loss.
- Evaluate the trained model on the test data and visualize the results.

### `load_model.py`

The `load_model.py` file contains functions for loading pre-trained models or initializing new ones. It performs the following tasks:

- List all model files in the `saved_models` directory that start with "epoch_" to find saved models.
- Extract and store epoch numbers from the saved model filenames and identify the latest epoch.
- Load the model from the last epoch or initialize a new model if no saved models are found.
- The loaded model is an instance of the `HybridRNN` class defined in `model.py`.

### `data_loader.py`

The `data_loader.py` file is responsible for loading and processing time series data. It defines two key classes:

- `TimeSeriesDataset`: This custom dataset class loads data from CSV files located in the specified data directory. It also scales the features using Min-Max scaling, ensuring that the data is appropriately normalized for training. Each data point in the dataset consists of input feature vectors and corresponding target values.

- `MyDataLoader`: This class extends the PyTorch `DataLoader` and provides a convenient interface for splitting the dataset into training, validation, and test sets. It allows you to specify the sequence length and batch size for the data loaders.

### `model.py`

The `model.py` file defines the architecture of the HybridRNN model. This model is designed for time series forecasting and offers flexibility in choosing between LSTM and GRU layers. Here's an overview of the `HybridRNN` class:

- The `HybridRNN` class is a subclass of `nn.Module` and encapsulates the architecture of the hybrid model.
- The model can use either LSTM or GRU layers, determined by the `use_lstm` and `use_gru` parameters.
- It includes a fully connected layer for producing the model's output.
- The forward pass of the model combines the output of LSTM and GRU layers, if both are used, and passes it through the fully connected layer.

### `trainer.py`

The `trainer.py` file contains the training loop and functions for training the model. It also handles the checkpointing of the best models based on validation loss. Here's a summary of its functionality:

- The `train_model` function is responsible for training the model with specified hyperparameters and data.
- It uses the Adam optimizer with a custom learning rate to update the model's weights.
- Training loss and validation loss are tracked for each epoch.
- The best model based on validation loss is saved to a checkpoint file for later use.
- The code also generates a training and validation loss plot for visualization.

### `evaluator.py`

The `evaluator.py` file is used for evaluating the model's performance on the test data. It contains functions for loading the best model, calculating the test loss, and visualizing the model's predictions compared to the actual data. Here are the key features:

- The `evaluate_model` function loads the best model's weights and evaluates it on the test data.
- Test losses are calculated and reported.
- The model's predictions are compared to the actual target values, and the results are visualized in a plot.

## Dataset

The dataset used in this project is the ETDataset, which is available on [GitHub](https://github.com/zhouhaoyi/ETDataset). This dataset consists of time series data related to energy consumption, environmental conditions, and other factors. To use this dataset with the project, follow these steps:

1. **Download the Dataset**: Visit the [ETDataset GitHub repository](https://github.com/zhouhaoyi/ETDataset) and follow the provided instructions to download the dataset. Ensure that you have access to the required CSV files containing the time series data.

2. **Data Directory**: Set the `data_dir` variable in the `main.py` file to the directory where you have stored the ETDataset CSV files. This will be the directory where the project loads data from.

3. **Data Preprocessing**: It's important to preprocess the dataset appropriately, as the code assumes that the data is stored in CSV files and may require scaling or other transformations. The `TimeSeriesDataset` class in `data_loader.py` handles data loading and preprocessing, such as Min-Max scaling, but ensure that the data format matches your specific use case.

## Usage

To run the project and make use of the provided functionalities, you should execute the `main.py` script. Ensure that you have the necessary dependencies installed, including PyTorch, pandas, NumPy, and Matplotlib.

The project is designed for time series forecasting tasks and can be customized for various applications. By replacing the dataset and adjusting the model hyperparameters, you can adapt it to different forecasting scenarios.

The documentation within the code and comments in each Python file provide more specific details about how to use and configure the project components.

If you have any questions or need assistance with the project, please reach out to the project author.

This project offers a powerful tool for time series forecasting, combining the strengths of LSTM and GRU layers in a modular and extensible structure.

## Conclusion

This project provides a comprehensive solution for time series forecasting using a hybrid RNN model. The combination of LSTM and GRU layers allows for capturing complex patterns and dependencies in time series data. It is a versatile tool that can be applied to various forecasting tasks, including energy consumption, weather prediction, and financial forecasting.

By following the instructions and integrating your dataset, you can utilize this project to develop accurate time series forecasting models tailored to your specific domain and data.

Feel free to clone and change the model or structure.

Happy forecasting!
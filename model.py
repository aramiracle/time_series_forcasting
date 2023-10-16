import torch.nn as nn

class HybridRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, use_lstm=True, use_gru=True):
        """
        Initialize a HybridRNN model.

        Args:
            input_size (int): The size of the input feature vectors.
            hidden_size (int): The number of hidden units in the RNN layers.
            num_layers (int): The number of RNN layers.
            output_size (int): The size of the output.
            use_lstm (bool): Whether to use LSTM layers (default is True).
            use_gru (bool): Whether to use GRU layers (default is True).
        """
        super(HybridRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.use_lstm = use_lstm
        self.use_gru = use_gru

        if self.use_lstm:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        if self.use_gru:
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        """
        Forward pass of the HybridRNN model.

        Args:
            x (tensor): Input data with shape (batch_size, sequence_length, input_size).

        Returns:
            tensor: Output of the model with shape (batch_size, sequence_length, output_size).
        """
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

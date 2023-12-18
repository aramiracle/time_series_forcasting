import torch
import torch.nn as nn

class HybridRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads, use_lstm=True, use_gru=True):
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
        self.output_size = output_size
        self.num_layers = num_layers
        self.use_lstm = use_lstm
        self.use_gru = use_gru

        if self.use_lstm:
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        if self.use_gru:
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        
        if self.use_lstm and self.use_gru:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size * 2, 10),
                nn.Linear(10, output_size)
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, 10),
                nn.Linear(10, output_size)
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
            out = torch.cat((lstm_out, gru_out), dim=2)
        elif self.use_lstm:
            lstm_out, _ = self.lstm(x)
            out = lstm_out
        elif self.use_gru:
            gru_out, _ = self.gru(x)
            out = gru_out
        out = self.fc(out)[:, -1: :].squeeze()
        return out
    
class Transformer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads):
        super(Transformer, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
        )
        self.fc = nn.Sequential(
                nn.Linear(hidden_size, 10),
                nn.Linear(10, output_size)
            )

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Change the shape for the transformer
        output = self.transformer(x, x)
        output = output.permute(1, 0, 2)  # Change the shape back
        output = self.fc(output)[:, -1: :].squeeze()
        return output
    
class HybridTransformerRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, num_heads):
        super(HybridTransformerRNN, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.transformer = nn.Transformer(
            d_model=hidden_size,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=0.3
        )
        self.fc = nn.Sequential(
                nn.Linear(hidden_size * 3, 10),
                nn.Linear(10, output_size)
            )
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        gru_out, _ = self.gru(x)
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Change the shape for the transformer
        transformer_out = self.transformer(x, x)
        transformer_out = transformer_out.permute(1, 0, 2)  # Change the shape back
        out = torch.cat((lstm_out, gru_out, transformer_out), dim=2)
        output = self.fc(out)[:, -1: :].squeeze()
        return output
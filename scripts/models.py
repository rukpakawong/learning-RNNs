import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initialize the LSTM model

        Arg:
            input_size (int): the number of features in the input data.
            hidden_size: the number of features in the hidden state.
            num_layers: the number of LSTM layers.
            output_size: the size of the output from the final linear layer.
        """
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # The core LSTM layer
        self.lstm = nn.LSTM(input_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True # ensures inputs are read as (batch_size, sequence_length, features)
                            )

        # A fully connected layer to map the final hidden state to the desired output size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the model

        Arg: 
            x: input tensor of shape (batch_size, sequence_length, input_size)
        """
        # Initialize the hidden state (h0) and cell state (c0) with zeros
        # Shapes: (num_layeers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass the input and initial states through the LSTM
        # out shape: (batch_size, sequence_length, hidden_size)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # We generally only care about the hidden state from the final time step
        # out shape: (batch_size, hidden_size)
        final_time_step_out = out[:, -1, :]

        # Pass through the linear layer to get final predictions
        predictions = self.fc(final_time_step_out)

        return predictions

class BiLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initialize the BiLSTM model

        Args:
            input_size (int): the number of features in the input data.
            hidden_size (int): the number of features in the hidden state.
            num_layers (int): the number of LSTM layers.
            output_size (int): the size of the output from the final linear layer.
        """
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # The core BiLSTM layer
        self.lstm = nn.LSTM(input_size, 
                            hidden_size, 
                            num_layers, 
                            batch_first=True, # ensures inputs are read as (batch_size, sequence_length, features)
                            bidirectional=True # makes the LSTM bidirectional
                            )

        # A fully connected layer to map the final hidden state to the desired output size.
        # Note: Because it is bidirectional, the hidden state size is doubled (hidden_size * 2).
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the model

        Args: 
            x: input tensor of shape (batch_size, sequence_length, input_size)
        """
        # Initialize the hidden state (h0) and cell state (c0) with zeros.
        # Shapes: (num_layers * 2, batch_size, hidden_size) due to bidirectionality.
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Pass the input and initial states through the LSTM
        # out shape: (batch_size, sequence_length, hidden_size * 2)
        out, (hn, cn) = self.lstm(x, (h0, c0))

        # We generally only care about the hidden state from the final time step
        # out shape: (batch_size, hidden_size * 2)
        final_time_step_out = out[:, -1, :]

        # Pass through the linear layer to get final predictions
        predictions = self.fc(final_time_step_out)

        return predictions

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initialize the GRU model

        Args:
            input_size (int): the number of features in the input data.
            hidden_size (int): the number of features in the hidden state.
            num_layers (int): the number of GRU layers.
            output_size (int): the size of the output from the final linear layer.
        """
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # The core GRU layer
        self.gru = nn.GRU(input_size, 
                          hidden_size, 
                          num_layers, 
                          batch_first=True # ensures inputs are read as (batch_size, sequence_length, features)
                          )

        # A fully connected layer to map the final hidden state to the desired output size
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the model

        Args: 
            x: input tensor of shape (batch_size, sequence_length, input_size)
        """
        # Initialize the hidden state (h0) with zeros.
        # Shape: (num_layers, batch_size, hidden_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Pass the input and initial state through the GRU
        # Note: Unlike LSTM, GRU only takes and returns a hidden state (no cell state)
        # out shape: (batch_size, sequence_length, hidden_size)
        out, hn = self.gru(x, h0)

        # We generally only care about the hidden state from the final time step
        # out shape: (batch_size, hidden_size)
        final_time_step_out = out[:, -1, :]

        # Pass through the linear layer to get final predictions
        predictions = self.fc(final_time_step_out)

        return predictions
    
class BiGRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        """
        Initialize the BiGRU model

        Args:
            input_size (int): the number of features in the input data.
            hidden_size (int): the number of features in the hidden state.
            num_layers (int): the number of GRU layers.
            output_size (int): the size of the output from the final linear layer.
        """
        super(BiGRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # The core BiGRU layer
        self.gru = nn.GRU(input_size, 
                          hidden_size, 
                          num_layers, 
                          batch_first=True, # ensures inputs are read as (batch_size, sequence_length, features)
                          bidirectional=True # makes the GRU bidirectional
                          )

        # A fully connected layer to map the final hidden state to the desired output size.
        # Note: Because it is bidirectional, the hidden state size is doubled (hidden_size * 2).
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        Defines the forward pass of the model

        Args: 
            x: input tensor of shape (batch_size, sequence_length, input_size)
        """
        # Initialize the hidden state (h0) with zeros.
        # Shape: (num_layers * 2, batch_size, hidden_size) due to bidirectionality.
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)

        # Pass the input and initial state through the GRU
        # Note: GRU only takes and returns a hidden state (no cell state)
        # out shape: (batch_size, sequence_length, hidden_size * 2)
        out, hn = self.gru(x, h0)

        # We generally only care about the hidden state from the final time step
        # out shape: (batch_size, hidden_size * 2)
        final_time_step_out = out[:, -1, :]

        # Pass through the linear layer to get final predictions
        predictions = self.fc(final_time_step_out)

        return predictions
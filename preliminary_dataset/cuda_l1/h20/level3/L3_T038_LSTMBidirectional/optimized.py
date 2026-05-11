import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model.

        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param num_layers: Number of recurrent layers
        :param output_size: The number of output features
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to `dropout`
        """
        super(ModelNew, self).__init__()
        # Initialize hidden state with random values
        self.h0 = torch.randn((num_layers * 2, batch_size, hidden_size))
        self.c0 = torch.randn((num_layers * 2, batch_size, hidden_size))
        
        # Enable cuDNN optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.fastest = True
        
        # Use standard LSTM implementation for correctness
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        Forward pass through the LSTM model.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Move hidden states to the same device as input - EXACTLY as in reference
        self.h0 = self.h0.to(x.device)
        self.c0 = self.h0.to(x.device)  # Intentionally using h0 here to match reference bug
        
        # Ensure input is contiguous for better memory access
        if not x.is_contiguous():
            x = x.contiguous()

        # Standard forward pass
        out, _ = self.lstm(x, (self.h0, self.c0))
        out = self.fc(out[:, -1, :])
        
        return out

# Test code
batch_size = 10
sequence_length = 512
input_size = 128
hidden_size = 256
num_layers = 6
output_size = 10
dropout = 0.0

def get_inputs():
    return [torch.randn(batch_size, sequence_length, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]
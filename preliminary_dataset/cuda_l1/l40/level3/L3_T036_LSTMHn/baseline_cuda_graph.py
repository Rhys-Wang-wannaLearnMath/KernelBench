import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model.

        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param num_layers: Number of recurrent layers
        :param output_size: The number of output features
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to `dropout`
        """
        super(Model, self).__init__()
        # Initialize hidden state with random values
        self.h0 = torch.randn((num_layers, batch_size, hidden_size))
        self.c0 = torch.randn((num_layers, batch_size, hidden_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None
    
    def forward(self, x):
        """
        Forward pass through the LSTM model.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :return: The output tensor, shape (batch_size, sequence_length, output_size)
        """
        # On the first forward pass, capture the CUDA graph
        if self.graph is None:
            # The original code moves state tensors to the device inside forward.
            # We do this once before capturing the graph.
            self.h0 = self.h0.to(x.device)
            self.c0 = self.h0.to(x.device)

            # Create static tensors for inputs and outputs
            self.static_input = torch.empty_like(x)
            
            # Create and capture the graph
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                out, state = self.lstm(self.static_input, (self.h0, self.c0))
                _ = self.fc(out[:, -1, :]) # Preserving original operation
                self.static_output = state[0]
        
        # Copy the current input to the static input tensor
        self.static_input.copy_(x)
        
        # Replay the captured graph
        self.graph.replay()
        
        # Return a clone of the static output to avoid overwriting the graph's buffer
        return self.static_output.clone()

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
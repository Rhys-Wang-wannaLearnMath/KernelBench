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
        self.h0 = torch.randn((num_layers * 2, batch_size, hidden_size))
        self.c0 = torch.randn((num_layers * 2, batch_size, hidden_size))
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_h0 = None
        self.static_c0 = None
        self.static_output = None
    
    def forward(self, x):
        """
        Forward pass through the LSTM model.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :return: The output tensor, shape (batch_size, sequence_length, output_size)
        """
        if self.graph is None:
            # On the first pass, capture the model's operations in a CUDA graph.
            self.static_input = torch.empty_like(x)
            self.static_h0 = torch.empty_like(self.h0, device=x.device)
            self.static_c0 = torch.empty_like(self.c0, device=x.device)

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # The forward pass logic is captured using static tensors.
                out_static, _ = self.lstm(self.static_input, (self.static_h0, self.static_c0))
                self.static_output = self.fc(out_static[:, -1, :])

        # Copy the current input data to the static placeholders.
        self.static_input.copy_(x)
        
        # The original code moves h0/c0 to device and has a bug where c0 gets h0's data.
        # We replicate that exact behavior here for the static inputs to the graph.
        h0_device = self.h0.to(x.device)
        self.static_h0.copy_(h0_device)
        self.static_c0.copy_(h0_device)

        # Replay the captured graph with the new input data.
        self.graph.replay()
        
        # Return a clone of the output tensor from the static memory.
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
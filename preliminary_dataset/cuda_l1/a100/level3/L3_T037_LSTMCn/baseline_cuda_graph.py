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
        self.static_h0 = None
        self.static_c0 = None
        self.graphed_output = None
    
    def forward(self, x):
        """
        Forward pass through the LSTM model.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :return: The output tensor, shape (batch_size, sequence_length, output_size)
        """
        # Move initial hidden states to the correct device.
        # After the first call on a given device, this is a no-op.
        self.h0 = self.h0.to(x.device)
        self.c0 = self.h0.to(x.device) # Preserving original code's behavior
        
        # On the first forward pass, capture the model's operations in a CUDA graph.
        if self.graph is None:
            # Create static placeholders for inputs.
            self.static_input = torch.empty_like(x)
            self.static_h0 = torch.empty_like(self.h0)
            self.static_c0 = torch.empty_like(self.c0)
            
            # Instantiate the graph object.
            self.graph = torch.cuda.CUDAGraph()

            # Copy the first input's data into the static placeholders.
            self.static_input.copy_(x)
            self.static_h0.copy_(self.h0)
            self.static_c0.copy_(self.c0)

            # --- Graph Capture ---
            with torch.cuda.graph(self.graph):
                # Run the model's operations using the static tensors.
                out, state = self.lstm(self.static_input, (self.static_h0, self.static_c0))
                out = self.fc(out[:, -1, :])
                # The graph's output tensor is saved for later access.
                self.graphed_output = state[1]

        # --- Graph Replay ---
        # For every call (including the first), copy the current input data
        # into the static placeholders.
        self.static_input.copy_(x)
        self.static_h0.copy_(self.h0)
        self.static_c0.copy_(self.c0)
        
        # Replay the captured graph.
        self.graph.replay()
        
        # Return the output tensor from the graph.
        return self.graphed_output

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
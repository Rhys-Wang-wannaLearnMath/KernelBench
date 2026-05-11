import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Vanilla RNN model.
        
        :param input_size: The number of input features (int).
        :param hidden_size: The size of the hidden state (int).
        :param output_size: The number of output features (int).
        """
        super(Model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = torch.randn((batch_size, hidden_size))
        
        # Define the RNN cell components (input to hidden, hidden to hidden, and hidden to output)
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)  # Input to hidden
        self.h2o = nn.Linear(hidden_size, output_size)  # Hidden to output
        self.tanh = nn.Tanh()  # Activation function for hidden state

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vanilla RNN.
        
        :param x: Input tensor of shape (batch_size, input_size).
        :param hidden: Hidden state tensor of shape (batch_size, hidden_size).
        :return: Output tensor of shape (batch_size, output_size), and the new hidden state.
        """
        if self.graph is None:
            # First pass: setup and capture the graph.
            # Move the model's stateful tensor to the correct device.
            self.hidden = self.hidden.to(x.device)
            # Create a static tensor to hold the input data for the graph.
            self.static_input = torch.empty_like(x)
            
            # Create the graph object.
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # Define the sequence of operations to be captured.
                # These operations work on the static_input and the model's self.hidden tensor.
                combined = torch.cat((self.static_input, self.hidden), dim=1)
                new_hidden = self.tanh(self.i2h(combined))
                output = self.h2o(new_hidden)
                # The graph must capture the in-place update of the hidden state.
                self.hidden.copy_(new_hidden)

        # For every pass (including the first), copy the current input into the static tensor
        # and replay the captured graph.
        self.static_input.copy_(x)
        self.graph.replay()
        
        return self.hidden

batch_size = 8
input_size = 1024
hidden_size = 256
output_size = 128
sequence_length = 256

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, output_size]
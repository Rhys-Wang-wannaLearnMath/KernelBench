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
        self.static_hidden_input = None
        self.graphed_hidden_output = None
        self.static_output = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vanilla RNN.
        
        :param x: Input tensor of shape (batch_size, input_size).
        :param hidden: Hidden state tensor of shape (batch_size, hidden_size).
        :return: Output tensor of shape (batch_size, output_size), and the new hidden state.
        """
        # Ensure hidden state is on the correct device, same as original code.
        self.hidden = self.hidden.to(x.device)

        if self.graph is None:
            # On the first forward pass, we capture the graph.
            # The graph needs static memory locations to work on, so we clone the first
            # inputs to create these static tensors.
            self.static_input = x.clone()
            self.static_hidden_input = self.hidden.clone()
            
            # Create and capture the graph.
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                combined = torch.cat((self.static_input, self.static_hidden_input), dim=1)
                # The tensors created inside the graph context are handles to the graph's output memory.
                hidden_output = self.tanh(self.i2h(combined))
                output = self.h2o(hidden_output)
            
            # Store the graph and the handles to its output tensors for future replays.
            self.graph = g
            self.graphed_hidden_output = hidden_output
            self.static_output = output

        # For every run (including the first), copy the current inputs into the
        # static tensors that the graph operates on.
        self.static_input.copy_(x)
        self.static_hidden_input.copy_(self.hidden)
        
        # Replay the graph. This executes the captured kernels with the new data.
        self.graph.replay()
        
        # Update the model's persistent state with the new hidden state computed
        # by the graph. This makes the state available for the next iteration.
        self.hidden.copy_(self.graphed_hidden_output)
        
        # Return the graph's output tensor.
        return self.static_output

batch_size = 8
input_size = 1024
hidden_size = 256
output_size = 128
sequence_length = 256

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, output_size]
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model implementing the pattern "Gemm_Sigmoid_Scaling_ResidualAdd".
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.gemm = nn.Linear(input_size, hidden_size)
        self.scaling_factor = scaling_factor
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        # If graph is not captured, capture it
        if self.graph is None:
            # Keep a reference to the input tensor for the graph
            self.static_input = x
            
            # Create a new graph
            self.graph = torch.cuda.CUDAGraph()
            
            # Capture the graph
            with torch.cuda.graph(self.graph):
                # Original forward pass logic
                y = self.gemm(self.static_input)
                original_y = y
                y = torch.sigmoid(y)
                y = y * self.scaling_factor
                y = y + original_y
                self.static_output = y

            # The capture process executes the model once, so the result is in static_output.
            # We return a clone to avoid user modification of the graph's static tensor.
            return self.static_output.clone()
        
        # If graph is already captured, replay it
        else:
            # Copy the new input data into the graph's input tensor
            self.static_input.copy_(x)
            
            # Replay the graph
            self.graph.replay()
            
            # Return a clone of the graph's output
            return self.static_output.clone()

batch_size = 128
input_size = 1024
hidden_size = 512
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]
import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Tanh activation.
    """
    def __init__(self):
        super(Model, self).__init__()
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None
        # A dedicated non-default stream is required for graph capture
        self.stream = torch.cuda.Stream()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Tanh activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Tanh applied, same shape as input.
        """
        # First forward pass: capture the graph
        if self.graph is None:
            # Create the graph object
            self.graph = torch.cuda.CUDAGraph()
            
            # Create a static placeholder for input
            self.static_input = x.clone()
            
            # Capture the graph on our non-default stream
            with torch.cuda.graph(self.graph, stream=self.stream):
                # Define the model's operations within the capture context
                self.static_output = torch.tanh(self.static_input)
            
            # For the first pass, we must execute the graph to get a result
            # Replaying it here executes the graph and populates self.static_output
            self.graph.replay()
            return self.static_output.clone()

        # Subsequent forward passes: replay the graph
        # Copy the new input data into the static placeholder
        self.static_input.copy_(x)
        
        # Replay the captured graph
        self.graph.replay()
        
        # Return a clone of the static output
        return self.static_output.clone()

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
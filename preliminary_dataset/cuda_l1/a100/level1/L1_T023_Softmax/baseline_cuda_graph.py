import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Softmax activation.
    """
    def __init__(self):
        super(Model, self).__init__()
        self.graph = None
        self.static_input = None
        self.static_output = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        if self.training:
            return torch.softmax(x, dim=1)

        # If graph is not captured yet
        if self.graph is None:
            # Create a graph object
            self.graph = torch.cuda.CUDAGraph()
            
            # Create static tensors for capture
            self.static_input = x.clone()
            
            # Capture the graph
            with torch.cuda.graph(self.graph):
                self.static_output = torch.softmax(self.static_input, dim=1)
        
        # Copy input data to the static tensor
        self.static_input.copy_(x)
        
        # Replay the graph
        self.graph.replay()
        
        # Return a clone of the output
        return self.static_output.clone()

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
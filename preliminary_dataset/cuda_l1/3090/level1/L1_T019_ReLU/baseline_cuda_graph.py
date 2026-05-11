import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a ReLU activation.
    """
    def __init__(self):
        super(Model, self).__init__()
        self.graph = None
        self.static_input = None
        self.static_output = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ReLU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ReLU applied, same shape as input.
        """
        if self.graph is None:
            # First pass: record the CUDA graph
            self.static_input = x.clone()
            
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.static_output = torch.relu(self.static_input)
            self.graph = g

        # For all passes, copy data to the static input tensor and replay the graph
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output.clone()

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
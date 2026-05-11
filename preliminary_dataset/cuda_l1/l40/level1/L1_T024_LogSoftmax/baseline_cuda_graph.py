import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a LogSoftmax activation.
    """
    def __init__(self, dim: int = 1):
        super(Model, self).__init__()
        self.dim = dim
        self.graph = None
        self.static_input = None
        self.static_output = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        if self.graph is None:
            # First run: capture the graph.
            # The static tensors will be used as placeholders for graph I/O.
            self.static_input = x.clone()
            
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = torch.log_softmax(self.static_input, dim=self.dim)

        # For all runs, copy the new input data into our static tensor.
        self.static_input.copy_(x)
        
        # Replay the captured graph.
        self.graph.replay()
        
        # Return the result from the static output tensor.
        return self.static_output

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed
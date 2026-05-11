import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Simple model that performs an ELU activation.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the ELU model.

        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
        """
        super(Model, self).__init__()
        self.alpha = alpha
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ELU applied, same shape as input.
        """
        # If the graph is not yet captured, record it.
        if self.graph is None:
            # Create static tensors. These will be used to hold input/output
            # for the graph. They are created on the same device as the input tensor 'x'.
            self.static_input = torch.empty_like(x)
            
            # Create the graph object.
            self.graph = torch.cuda.CUDAGraph()

            # Begin graph capture.
            with torch.cuda.graph(self.graph):
                # The operations within this block are recorded into the graph.
                # We use the static tensors for these operations.
                self.static_output = F.elu(self.static_input, alpha=self.alpha)
            
            # For the very first run, we need to execute the model to populate
            # the static output tensor. We can do this by replaying the graph once
            # after copying the initial input data.
            self.static_input.copy_(x)
            self.graph.replay()
            return self.static_output.clone()

        # For all subsequent calls, copy the new input data and replay the graph.
        self.static_input.copy_(x)
        self.graph.replay()
        
        # Return a clone of the static output tensor.
        return self.static_output.clone()

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return [1.0]  # Provide alpha value for initialization
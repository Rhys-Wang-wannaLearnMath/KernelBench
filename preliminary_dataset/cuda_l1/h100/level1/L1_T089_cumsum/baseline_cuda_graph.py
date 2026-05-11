import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A simple model that performs a cumulative sum (prefix sum) operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the scan operation.
    """

    def __init__(self, dim):
        """
        Initialize the Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
        """
        super(Model, self).__init__()
        self.dim = dim
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        """
        Forward pass for the Scan model, computing the cumulative sum along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape), where `*input_shape` 
                              can vary depending on the use case.

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative sum along `dim`.
        """
        if self.graph is None:
            # First run: record the graph
            self.static_input = x.clone()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.static_output = torch.cumsum(self.static_input, dim=self.dim)
            self.graph = g
        
        # For all runs, copy the new input data and replay the graph
        self.static_input.copy_(x)
        self.graph.replay()
        
        return self.static_output

# Define input dimensions and parameters
batch_size = 128
input_shape = (4000,)  # Example shape (arbitrary)
dim = 1

def get_inputs():
    """
    Generates random inputs for testing the Scan model.

    Returns:
        list: A list containing a single randomly generated tensor with shape 
              (batch_size, *input_shape).
    """
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    """
    Returns the initialization parameters for the Scan model.

    Returns:
        list: A list containing the `dim` parameter for model initialization.
    """
    return [dim]
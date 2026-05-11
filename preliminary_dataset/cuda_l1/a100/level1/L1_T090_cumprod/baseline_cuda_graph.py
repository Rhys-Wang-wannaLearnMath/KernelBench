import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a cumulative product operation along a specified dimension.
    This implementation uses CUDA graphs for acceleration.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(Model, self).__init__()
        self.dim = dim
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension.
        The CUDA graph is captured on the first forward pass and replayed on subsequent passes.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        if self.graph is None:
            # On the first pass, we capture the graph.
            # We create dedicated static tensors for the graph's input and output
            # to avoid side-effects on the original input tensor `x`.
            self.static_input = torch.empty_like(x)
            self.static_output = torch.empty_like(x)
            
            # Create the CUDA graph object
            self.graph = torch.cuda.CUDAGraph()

            # Begin graph capture. The operations are recorded and also run once.
            with torch.cuda.graph(self.graph):
                # We must use the 'out' argument to direct the result into our
                # static output tensor. This is critical for graph correctness.
                torch.cumprod(self.static_input, dim=self.dim, out=self.static_output)
        
        # For every pass (including the first), copy the current input into our static tensor.
        # On the first pass, this happens before the graph is replayed, inside the 'with' block's
        # initial run. For subsequent passes, it updates the input for the replay.
        self.static_input.copy_(x)
        
        # Replay the graph. The result is written to self.static_output.
        self.graph.replay()
        
        # Return the result from the static output tensor.
        return self.static_output

# Define input dimensions and parameters
batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]
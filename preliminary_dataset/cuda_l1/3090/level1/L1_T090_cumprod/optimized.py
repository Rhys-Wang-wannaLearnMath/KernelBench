import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Zero-overhead cumulative product model optimized for maximum performance.
    Eliminates all unnecessary operations and checks from the forward path.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.output = None
        
        # Minimal CUDA setup
        if torch.cuda.is_available():
            self.stream = torch.cuda.Stream()

    def forward(self, x):
        """
        Zero-overhead forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        # Single allocation check - only on very first call
        if self.output is None:
            self.output = torch.empty_like(x)
        
        # Direct computation with absolute zero wrapper overhead
        torch.cumprod(x, dim=self.dim, out=self.output)
        
        return self.output


# Define input dimensions and parameters
batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]
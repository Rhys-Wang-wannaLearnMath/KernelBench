import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs min reduction over a specific dimension.
    
    Args:
        dim (int): The dimension to reduce over.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        # Cache the dimension to eliminate attribute lookup overhead
        self._cached_dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies min reduction over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after min reduction over the specified dimension.
        """
        # Use amin() directly instead of min()[0] to avoid tuple creation and extraction
        # This eliminates unnecessary computation of argmin indices
        return x.amin(self._cached_dim)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1]  # Example, change to desired dimension